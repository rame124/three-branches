"""Three Branches - Season 3 village-life agent.

Goals:
- Use very lightweight local cell steering to stay within Season 3 timing limits.
- Give villagers stable jobs without making work their whole personality.
- Give every odd-numbered villager a one-time morning crop chore near home.
- Alternate work with short wandering, sweeping, resting, and social breaks.
- React to the player and occasionally interact with other villagers.
- Build the complete navigation graph during reset and reuse it for all movement.
- Provide lightweight debug output in both stdout and in-game chat.
"""

from collections import deque

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

DEBUG = True
DEBUG_CHAT = True

WALK_SPEED = 0.78
SLOW_APPROACH_SPEED = 0.20
DOORWAY_WORK_SPEED = 0.38
STUCK_LIMIT = 5
INTERIOR_DOOR_STUCK_LIMIT = 10
HOME_STUCK_LIMIT = 3
HOME_YIELD_TICKS = 6

GENERIC_WORK_TICKS = 18

BREAK_MIN_TICKS = 35
BREAK_MAX_TICKS = 80
SOCIAL_MIN_TICKS = 20
SOCIAL_MAX_TICKS = 45
SWEEP_MIN_TICKS = 20
SWEEP_MAX_TICKS = 45
REST_MIN_TICKS = 20
REST_MAX_TICKS = 45

PLAYER_GREETING_MIN_DELAY = 180
PLAYER_GREETING_MAX_DELAY = 360

# Season 3's actual prop vocabulary (from the in-game debug dump).
CROP_WORDS = ("plot",)
SHRINE_WORDS = ("shrine",)
STALL_WORDS = ("stall",)
WORKBENCH_WORDS = ("repair bench",)
HEARTH_WORDS = ("hearth",)
PUMP_WORDS = ("pump",)
LANTERN_WORDS = ("lantern",)
BOARD_WORDS = ("board",)
BELL_WORDS = ("bell",)
BENCH_WORDS = ("bench",)

# Navigation is intentionally built completely during reset().
# This causes a short initial load pause, but keeps gameplay deterministic and
# avoids leaving villagers asleep or spreading expensive setup over live ticks.


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _cell_key(cell):
    return cell["x"], cell["y"]


def _cell_dict(key):
    return {"x": key[0], "y": key[1]}


def _cell_centre(observation, cell):
    frame = layout.frame(observation)
    size = frame.get("cell_size", 1.0)
    return {
        "x": (cell["x"] + 0.5) * size,
        "y": (cell["y"] + 0.5) * size,
    }


def _seat_number(player_id):
    try:
        return int(str(player_id).split("_")[-1])
    except (TypeError, ValueError):
        return 999


def _normalise_text(value):
    return (
        str(value or "")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def _prop_matches(prop, words):
    prop_type = _normalise_text(prop.get("type", ""))
    return any(_normalise_text(word) in prop_type for word in words)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """Role-based villager with morning crop duty and work/leisure cycles."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        self.seed = seed
        self.rng = me.rng(observation, seed)
        self.player_id = me.player_id(observation)
        self.seat = _seat_number(self.player_id)
        self.home = me.home(observation)

        # Complete navigation graph. All expensive map discovery happens once in
        # reset(); live gameplay only searches this cached graph.
        frame = layout.frame(observation)
        self.nav_cells_x = frame["cells_x"]
        self.nav_cells_y = frame["cells_y"]
        self.nav_graph = {}
        self.nav_expanded = set()
        self.nav_outdoor = set()
        self.previous_cell = None
        self.wake_tick = 0
        self.home_route = []

        start_cell = layout.cell_at(observation, me.position(observation))
        self.nav_start_cell = _cell_key(start_cell) if start_cell is not None else None

        # Cache a few exterior cells immediately outside our doorway. These are
        # cheap to identify and give the cached BFS an explicit way out of home.
        self.home_exit_cells = self._find_home_exit_cells(observation)

        # Cache static props once.
        self.all_props = list(props.all(observation))
        self.crop_plots = [
            prop for prop in self.all_props
            if _prop_matches(prop, CROP_WORDS)
        ]
        self.shrines = [
            prop for prop in self.all_props
            if _prop_matches(prop, SHRINE_WORDS)
        ]
        self.stalls = [
            prop for prop in self.all_props
            if _prop_matches(prop, STALL_WORDS)
        ]
        self.workbenches = [
            prop for prop in self.all_props
            if _prop_matches(prop, WORKBENCH_WORDS)
        ]
        self.hearths = [
            prop for prop in self.all_props
            if _prop_matches(prop, HEARTH_WORDS)
        ]
        self.pumps = [
            prop for prop in self.all_props
            if _prop_matches(prop, PUMP_WORDS)
        ]
        self.lanterns = [
            prop for prop in self.all_props
            if _prop_matches(prop, LANTERN_WORDS)
        ]
        self.boards = [
            prop for prop in self.all_props
            if _prop_matches(prop, BOARD_WORDS)
        ]
        self.bells = [
            prop for prop in self.all_props
            if _prop_matches(prop, BELL_WORDS)
        ]
        self.benches = [
            prop for prop in self.all_props
            if _prop_matches(prop, BENCH_WORDS)
        ]

        # Stable secondary role. Five roles means a cast of ten naturally
        # produces two villagers per job. Crop duty is separate from this.
        self.role = self._choose_role()
        # Roles own small task pools rather than a single prop type.
        if self.role == "shrine":
            self.role_props = self.shrines + self.pumps
        elif self.role == "stall":
            self.role_props = self.stalls + self.boards
        elif self.role == "workbench":
            self.role_props = self.workbenches + self.pumps
        elif self.role == "hearth":
            self.role_props = self.hearths + self.pumps
        else:
            # Groundskeepers alternate between lanterns/bell work and
            # free-form sweeping.
            self.role_props = self.lanterns + self.bells

        # Simple morning crop rule: every odd-numbered villager tends the
        # nearest plot to their home once, then resumes their normal job.
        self.crop_duty = (self.seat % 2 == 1) and bool(self.crop_plots)
        self.home_crop = (
            self._nearest_home_crop(observation)
            if self.crop_duty
            else None
        )
        self.crop_duty = self.home_crop is not None
        self.crop_finished = not self.crop_duty

        # Rebuild the complete Season 1-style navigation graph now. Nothing in
        # act() is allowed to discover or expand navigation later.
        self._build_reset_navigation(observation)

        # Activity state.
        self.current_activity = None
        self.planned_activity = None
        self.last_role_prop_id = None

        # Movement state.
        self.last_was_move = False
        self.stuck_ticks = 0
        self.home_stuck_ticks = 0
        self.home_yield_until = 0

        # Visitor interaction state.
        now = day.tick(observation)
        self.next_player_greeting = now + self.rng.randint(
            PLAYER_GREETING_MIN_DELAY,
            PLAYER_GREETING_MAX_DELAY,
        )

        # Debug state. chat() will emit this once in the in-game message panel.
        self.debug_chat_sent = False
        self.debug_prop_types_sent = False
        self.debug_chat_text = self._build_debug_summary()

        self._debug(
            f"START | home={self.home} | role={self.role} | "
            f"crop_duty={self.crop_duty} | "
            f"navigation=full_reset_graph:{len(self.nav_graph)} | wake_tick={self.wake_tick}"
        )

        if self.home_crop is not None:
            self._debug(
                f"MORNING CROP | id={self.home_crop['id']} | "
                f"type={self.home_crop.get('type')} | "
                f"cell={self.home_crop['cell']}"
            )

        if self.role_props:
            self._debug(
                "ROLE PROPS | "
                + ", ".join(
                    f"{prop['id']}:{prop.get('type')}"
                    for prop in self.role_props
                )
            )
        else:
            self._debug("ROLE PROPS | none")

        # Morning crop responsibility comes first, but is not the permanent job.
        if self.crop_duty:
            self.current_activity = self._make_work_activity(
                self.home_crop,
                "crop",
                observation,
            )
        else:
            self.current_activity = self._make_role_work_activity(observation)
            if self.current_activity is None:
                self.current_activity = self._make_break_activity(observation)

        self._debug_activity("INITIAL", self.current_activity)

    # ======================================================================
    # Debugging
    # ======================================================================

    def _debug(self, message):
        if DEBUG:
            print(f"[{self.player_id} | {self.role}] {message}", flush=True)

    def _build_debug_summary(self):
        crop_text = "none"
        if self.home_crop is not None:
            crop_text = (
                f"{self.home_crop.get('type')}@"
                f"{self.home_crop.get('cell')}"
            )
        return (
            f"DEBUG {self.player_id}: role={self.role}, home={self.home}, "
            f"crop_duty={self.crop_duty}, crop={crop_text}, "
            f"nav=reset_cache:{len(self.nav_expanded)}, wake_tick={self.wake_tick}"
        )

    def _debug_activity(self, prefix, activity):
        if not DEBUG or activity is None:
            return

        kind = activity.get("kind", "unknown")

        if kind == "work":
            prop = activity.get("prop")
            purpose = activity.get("purpose", "unknown")
            if prop is not None:
                self._debug(
                    f"{prefix} ACTIVITY | work:{purpose} | "
                    f"{prop.get('type')} | id={prop.get('id')} | "
                    f"cell={prop.get('cell')} | "
                    f"route={len(activity.get('route', []))}"
                )
            else:
                self._debug(f"{prefix} ACTIVITY | work:{purpose} | NO PROP")

        elif kind in ("wander", "sweep"):
            self._debug(
                f"{prefix} ACTIVITY | {kind} | "
                f"target={activity.get('target_cell')} | "
                f"route={len(activity.get('route', []))}"
            )
        elif kind == "social":
            self._debug(f"{prefix} ACTIVITY | social")
        elif kind == "rest":
            self._debug(f"{prefix} ACTIVITY | rest")
        else:
            self._debug(f"{prefix} ACTIVITY | {kind}")

    def chat(self, inbox):
        """Low-volume in-game diagnostics."""
        if not DEBUG_CHAT:
            return []

        if not self.debug_chat_sent:
            self.debug_chat_sent = True
            return [{"to": None, "text": self.debug_chat_text}]

        # Only player_1 reports the static prop vocabulary so the panel is not
        # flooded with ten identical copies. This is especially useful when a
        # Season changes prop type tokens.
        if self.seat == 1 and not self.debug_prop_types_sent:
            self.debug_prop_types_sent = True
            unique_types = sorted({
                _normalise_text(prop.get("type", ""))
                for prop in self.all_props
                if prop.get("type") is not None
            })
            text = "DEBUG PROP TYPES: " + ", ".join(unique_types)
            return [{"to": None, "text": text[:500]}]

        return []

    # ======================================================================
    # Static setup
    # ======================================================================

    def _in_bounds(self, key):
        return (
            0 <= key[0] < self.nav_cells_x
            and 0 <= key[1] < self.nav_cells_y
        )

    def _find_home_exit_cells(self, observation):
        """Find a few walkable exterior cells immediately beyond our door."""
        if self.home == "none":
            return []

        doorway = layout.doorway(observation, self.home)
        if doorway is None:
            return []

        door_cell = layout.cell_at(observation, doorway)
        if door_cell is None:
            return []

        dx0, dy0 = door_cell["x"], door_cell["y"]
        candidates = []

        for radius in (1, 2, 3):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    key = (dx0 + dx, dy0 + dy)
                    if not self._in_bounds(key):
                        continue
                    cell = _cell_dict(key)
                    if not layout.walkable(observation, cell):
                        continue
                    if layout.ground_at(observation, cell) == "interior":
                        continue
                    point = _cell_centre(observation, cell)
                    if layout.line_of_sight(observation, doorway, point):
                        candidates.append((geometry.distance(doorway, point), key))
            if candidates:
                break

        candidates.sort(key=lambda item: item[0])
        return [key for _, key in candidates[:4]]

    def _choose_role(self):
        """Assign a stable job strictly by seat number."""
        roles = ("shrine", "stall", "grounds", "workbench", "hearth")
        return roles[(self.seat - 1) % len(roles)]

    def _nearest_home_crop(self, observation):
        if not self.crop_plots:
            return None

        origin = None
        if self.home != "none":
            origin = layout.doorway(observation, self.home)
        if origin is None:
            origin = me.position(observation)

        return min(
            self.crop_plots,
            key=lambda prop: geometry.distance(
                origin,
                _cell_centre(observation, prop["cell"]),
            ),
        )

    def _purpose_for_prop(self, prop):
        prop_type = _normalise_text(prop.get("type", ""))
        if prop_type == "plot":
            return "crop"
        if prop_type == "shrine":
            return "shrine"
        if prop_type == "stall":
            return "stall"
        if prop_type == "repair bench":
            return "repair bench"
        if prop_type == "hearth":
            return "hearth"
        if prop_type == "pump":
            return "pump"
        if prop_type == "lantern":
            return "lantern"
        if prop_type == "board":
            return "board"
        if prop_type == "bell":
            return "bell"
        if prop_type == "bench":
            return "bench"
        return prop_type or "generic"

    def _build_reset_navigation(self, observation):
        """Build the complete village navigation graph during reset().

        This intentionally mirrors the reliable Season 1 approach: every map
        cell is inspected up front and every legal cardinal movement edge is
        cached. No navigation discovery is permitted during live gameplay.

        Keeping the current Season 3 role/task code unchanged makes this a
        clean diagnostic: if movement still fails with a complete graph, the
        remaining problem is in activity/goal logic rather than map knowledge.
        """
        self.nav_graph = {}
        self.nav_expanded = set()
        self.nav_outdoor = set()

        # First cache every walkable cell. This gives us a complete set of graph
        # nodes before edges are connected and avoids route failures caused by
        # partially known regions.
        for y in range(self.nav_cells_y):
            for x in range(self.nav_cells_x):
                key = (x, y)
                cell = _cell_dict(key)
                self.nav_expanded.add(key)

                if not layout.walkable(observation, cell):
                    continue

                self.nav_graph[key] = []
                if layout.ground_at(observation, cell) != "interior":
                    self.nav_outdoor.add(key)

        # Then connect every legal cardinal edge. We ask can_step from each
        # source cell just as the original Season 1 graph did, so any directional
        # movement rules are preserved rather than assuming all edges are
        # automatically symmetric.
        for current in list(self.nav_graph.keys()):
            start_cell = _cell_dict(current)
            neighbours = self.nav_graph[current]

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                other = (current[0] + dx, current[1] + dy)
                if other not in self.nav_graph:
                    continue

                if layout.can_step(observation, start_cell, _cell_dict(other)):
                    neighbours.append(other)

        edge_count = sum(len(v) for v in self.nav_graph.values())
        self._debug(
            f"RESET NAV FULL | nodes={len(self.nav_graph)} | "
            f"directed_edges={edge_count}"
        )

    # ======================================================================
    # Persistent cached navigation
    # ======================================================================

    def _activity_goals(self, observation, activity):
        if activity is None:
            return []
        kind = activity.get("kind")
        if kind == "work":
            return self._approach_cells(observation, activity["prop"])
        if kind in ("wander", "sweep"):
            target = activity.get("target_cell")
            return [target] if target is not None else []
        return []

    def _discover_cell(self, observation, current):
        """Discover one cell's legal cardinal edges exactly once."""
        if current in self.nav_expanded:
            return self.nav_graph.get(current, [])

        if not self._in_bounds(current):
            return []

        start_cell = _cell_dict(current)
        if not layout.walkable(observation, start_cell):
            self.nav_graph.setdefault(current, [])
            self.nav_expanded.add(current)
            return []

        neighbours = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = (current[0] + dx, current[1] + dy)
            if not self._in_bounds(other):
                continue
            if layout.can_step(observation, start_cell, _cell_dict(other)):
                neighbours.append(other)
                self.nav_graph.setdefault(other, [])
                if current not in self.nav_graph[other]:
                    self.nav_graph[other].append(current)

        self.nav_graph[current] = neighbours
        self.nav_expanded.add(current)

        if layout.ground_at(observation, start_cell) != "interior":
            self.nav_outdoor.add(current)

        return neighbours

    def _expand_navigation(self, observation, start, budget):
        """Expand at most ``budget`` new cached cells around ``start``.

        Already-known cells are traversed for free. Only cells whose edges have
        never been inspected count against the budget, so knowledge accumulates
        instead of being recomputed by every route request.
        """
        if start is None or not self._in_bounds(start) or budget <= 0:
            return 0

        queue = deque([start])
        seen = {start}
        expanded = 0

        while queue and expanded < budget:
            current = queue.popleft()
            was_new = current not in self.nav_expanded
            neighbours = self._discover_cell(observation, current)
            if was_new:
                expanded += 1

            for neighbour in neighbours:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)

        return expanded

    def _cached_neighbours(self, observation, current):
        """Return neighbours already discovered during reset()."""
        return self.nav_graph.get(current, [])

    def _route_in_cache(self, start, goals):
        """BFS over only the graph already discovered by this villager."""
        goal_set = set(goals)
        if not goal_set or start in goal_set:
            return []
        if start not in self.nav_graph:
            return []

        queue = deque([start])
        previous = {start: None}
        found = None

        while queue:
            current = queue.popleft()
            for neighbour in self.nav_graph.get(current, ()):
                if neighbour in previous:
                    continue
                previous[neighbour] = current
                if neighbour in goal_set:
                    found = neighbour
                    queue.clear()
                    break
                queue.append(neighbour)

        if found is None:
            return []

        route = []
        current = found
        while current != start:
            route.append(current)
            current = previous[current]
        route.reverse()
        return route

    def _frontier_toward_goals(self, goals):
        """Pick a discovered frontier cell that points toward a destination.

        This lets the personal map grow outward toward real tasks instead of
        repeatedly filling in more cells around home. No layout queries happen
        here; it only inspects the already-cached coordinates.
        """
        if not self.nav_graph or not goals:
            return None

        goal_list = list(goals)
        best = None
        best_distance = None

        for cell in self.nav_graph:
            # A frontier is a known cell bordering at least one cell whose
            # movement edges have not been inspected yet.
            is_frontier = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (cell[0] + dx, cell[1] + dy)
                if self._in_bounds(neighbour) and neighbour not in self.nav_expanded:
                    is_frontier = True
                    break

            if not is_frontier:
                continue

            distance = min(
                abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
                for goal in goal_list
            )

            if best_distance is None or distance < best_distance:
                best = cell
                best_distance = distance

        return best

    def _route_with_expansion(self, observation, start, goals):
        """Route only through the reset-time cache. Never expand during play."""
        return self._route_in_cache(start, set(goals))

    def _follow_activity_route(self, observation, activity, expression="none"):
        """Follow a persistent cached route and expand knowledge only if needed."""
        goals = self._activity_goals(observation, activity)
        if not goals:
            return None

        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)
        if here_cell is None:
            self.last_was_move = False
            return action.stand(me.heading(observation), expression)

        current = _cell_key(here_cell)
        if current in goals:
            activity["route"] = []
            self.last_was_move = False
            return None

        if self.last_was_move:
            if me.moved(observation) < 0.02:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0

        route = activity.setdefault("route", [])
        while route and route[0] == current:
            route.pop(0)

        # If another character displaced us, only invalidate when the cached
        # graph says the next step is no longer adjacent to our current cell.
        if route and route[0] not in self._cached_neighbours(observation, current):
            route.clear()

        purpose = activity.get("purpose")

        # Workbench/hearth jobs sometimes require re-entering a house through a
        # narrow doorway.  Re-planning too aggressively while physically in the
        # threshold can make an agent oscillate in the frame, so give those
        # transitions a little more time before invalidating the route.
        doorway_transition = False
        if route and purpose in ("workbench", "hearth"):
            current_ground = layout.ground_at(observation, _cell_dict(current))
            next_ground = layout.ground_at(observation, _cell_dict(route[0]))
            doorway_transition = current_ground != next_ground

        stuck_limit = (
            INTERIOR_DOOR_STUCK_LIMIT if doorway_transition else STUCK_LIMIT
        )
        if self.stuck_ticks >= stuck_limit:
            self.stuck_ticks = 0
            route.clear()

        if not route:
            route.extend(
                self._route_with_expansion(
                    observation,
                    current,
                    goals,
                )
            )

        if not route:
            self.last_was_move = False
            return action.stand(me.heading(observation), expression)

        next_key = route[0]
        target_key = next_key
        move_speed = WALK_SPEED

        if purpose in ("workbench", "hearth"):
            current_ground = layout.ground_at(observation, _cell_dict(current))
            next_ground = layout.ground_at(observation, _cell_dict(next_key))

            if current_ground != next_ground:
                move_speed = DOORWAY_WORK_SPEED

                # When entering a building, look one cell through the doorway if
                # the BFS path continues straight.  This prevents the doorway
                # threshold itself from becoming a stop point where the agent
                # can rotate back and forth.
                if (
                    current_ground != "interior"
                    and next_ground == "interior"
                    and len(route) >= 2
                ):
                    after_key = route[1]
                    dx1 = next_key[0] - current[0]
                    dy1 = next_key[1] - current[1]
                    dx2 = after_key[0] - next_key[0]
                    dy2 = after_key[1] - next_key[1]
                    if (dx1, dy1) == (dx2, dy2):
                        target_key = after_key

        target = _cell_centre(observation, _cell_dict(target_key))
        self.last_was_move = True
        return action.walk(
            geometry.heading_to(here, target),
            move_speed,
            expression,
        )

    # ======================================================================
    # Prop helpers
    # ======================================================================

    def _approach_cells(self, observation, prop):
        x = prop["cell"]["x"]
        y = prop["cell"]["y"]

        cardinal = []
        for candidate in (
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)
        ):
            if not self._in_bounds(candidate):
                continue
            if layout.walkable(observation, _cell_dict(candidate)):
                cardinal.append(candidate)

        if cardinal:
            return cardinal

        diagonal = []
        for candidate in (
            (x + 1, y + 1), (x + 1, y - 1),
            (x - 1, y + 1), (x - 1, y - 1),
        ):
            if not self._in_bounds(candidate):
                continue
            if layout.walkable(observation, _cell_dict(candidate)):
                diagonal.append(candidate)
        return diagonal

    def _prop_position(self, observation, prop):
        return _cell_centre(observation, prop["cell"])

    def _state_text(self, observation, prop):
        state = props.state(observation, prop["id"])
        if state is None:
            return None

        return (
            str(state)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

    def _work_status(self, observation, prop, purpose):
        """Return 'needs', 'done', or 'unknown'."""
        state = self._state_text(observation, prop)
        if state is None:
            return "unknown"

        if purpose == "crop":
            if any(
                word in state
                for word in (
                    "overgrown",
                    "untended",
                    "unattended",
                    "needs tending",
                    "needs attention",
                )
            ):
                return "needs"
            return "done"

        if purpose == "stall":
            if "closed" in state:
                return "needs"
            if "open" in state and "closed" not in state:
                return "done"
            return "unknown"

        if purpose == "shrine":
            if any(
                word in state
                for word in (
                    "untended",
                    "unattended",
                    "overgrown",
                    "unlit",
                    "needs tending",
                    "needs attention",
                )
            ):
                return "needs"

            if any(
                word in state
                for word in ("tended", "attended", "lit")
            ):
                return "done"

        if purpose == "hearth":
            if any(
                word in state
                for word in (
                    "unlit",
                    "cold",
                    "out",
                    "needs tending",
                    "needs attention",
                )
            ):
                return "needs"
            if any(
                word in state
                for word in (
                    "lit",
                    "burning",
                    "warm",
                    "tended",
                )
            ):
                return "done"

        if purpose == "lantern":
            if any(word in state for word in ("unlit", "off", "dark")):
                return "needs"
            if any(word in state for word in ("lit", "on", "burning")):
                return "done"
            return "unknown"

        # These are activity stations rather than simple on/off props.
        if purpose in ("repair bench", "pump", "board", "bell", "bench"):
            return "unknown"

        return "unknown"

    # ======================================================================
    # Activity creation
    # ======================================================================

    def _make_work_activity(self, prop, purpose, observation):
        if prop is None:
            return None

        activity = {
            "kind": "work",
            "prop": prop,
            "purpose": purpose,
            "route": [],
            "use_ticks": 0,
            "used_once": False,
        }
        return activity

    def _choose_role_prop(self, observation):
        if not self.role_props:
            return None

        here = me.position(observation)
        candidates = []

        here_cell = layout.cell_at(observation, here)
        if here_cell is None:
            return None

        for prop in self.role_props:
            # Static prop locations are standing knowledge, so jobs do not
            # have to already be inside the personal navigation cache. A
            # distant job becomes the reason to expand the cache toward that
            # part of the village.
            distance = geometry.distance(
                here,
                self._prop_position(observation, prop),
            )
            repeat_penalty = 8.0 if prop["id"] == self.last_role_prop_id else 0.0
            candidates.append((distance + repeat_penalty, prop))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        nearby = [prop for _, prop in candidates[:3]]
        return self.rng.choice(nearby)

    def _make_role_work_activity(self, observation):
        if self.role == "grounds" and self.rng.random() < 0.45:
            return self._make_sweep_activity(observation)

        prop = self._choose_role_prop(observation)
        if prop is None:
            return None

        return self._make_work_activity(
            prop,
            self._purpose_for_prop(prop),
            observation,
        )

    def _random_outdoor_target(self, observation):
        """Choose from already-known outdoor navigation cells."""
        here_cell = layout.cell_at(observation, me.position(observation))
        if here_cell is None:
            return None
        here_key = _cell_key(here_cell)

        candidates = [cell for cell in self.nav_outdoor if cell != here_key]
        if not candidates:
            legal = self._cached_neighbours(observation, here_key)
            return self.rng.choice(legal) if legal else None

        # Prefer the farther half of known outdoor space so wandering naturally
        # pushes villagers toward the edge of their personal map instead of
        # orbiting their house forever.
        sample = self.rng.sample(candidates, min(16, len(candidates)))
        sample.sort(
            key=lambda cell: abs(cell[0] - here_key[0]) + abs(cell[1] - here_key[1])
        )
        farther_half = sample[max(0, len(sample) // 2):]
        return self.rng.choice(farther_half or sample)

    def _make_wander_activity(self, observation):
        target = self._random_outdoor_target(observation)

        activity = {
            "kind": "wander",
            "target_cell": target,
            "route": [],
            "until": day.tick(observation) + self.rng.randint(
                BREAK_MIN_TICKS,
                BREAK_MAX_TICKS,
            ),
        }
        return activity

    def _make_sweep_activity(self, observation):
        target = self._random_outdoor_target(observation)

        activity = {
            "kind": "sweep",
            "target_cell": target,
            "route": [],
            "until": day.tick(observation) + self.rng.randint(
                SWEEP_MIN_TICKS,
                SWEEP_MAX_TICKS,
            ),
        }
        return activity

    def _make_rest_activity(self, observation):
        return {
            "kind": "rest",
            "until": day.tick(observation) + self.rng.randint(
                REST_MIN_TICKS,
                REST_MAX_TICKS,
            ),
        }

    def _make_social_activity(self, observation):
        return {
            "kind": "social",
            "until": day.tick(observation) + self.rng.randint(
                SOCIAL_MIN_TICKS,
                SOCIAL_MAX_TICKS,
            ),
            "target_id": None,
            "expression": self.rng.choice(("nod", "wave", "laugh")),
        }

    def _make_break_activity(self, observation):
        roll = self.rng.random()

        if self.role == "grounds":
            if roll < 0.35:
                return self._make_sweep_activity(observation)
            if roll < 0.65:
                return self._make_social_activity(observation)
            if roll < 0.90:
                return self._make_wander_activity(observation)
            return self._make_rest_activity(observation)

        if roll < 0.35:
            return self._make_social_activity(observation)
        if roll < 0.65:
            return self._make_wander_activity(observation)
        if roll < 0.85:
            return self._make_sweep_activity(observation)
        return self._make_rest_activity(observation)

    # ======================================================================
    # Plan the next route while standing still
    # ======================================================================

    def _prepare_next_break(self, observation):
        if self.planned_activity is not None:
            return
        self.planned_activity = self._make_break_activity(observation)
        self._debug_activity("PLANNED", self.planned_activity)

    def _prepare_next_work(self, observation):
        if self.planned_activity is not None:
            return

        if self.crop_duty and not self.crop_finished:
            self.planned_activity = self._make_work_activity(
                self.home_crop,
                "crop",
                observation,
            )
        else:
            self.planned_activity = self._make_role_work_activity(observation)

        if self.planned_activity is None:
            self.planned_activity = self._make_wander_activity(observation)

        self._debug_activity("PLANNED", self.planned_activity)

    def _advance_activity(self, observation, next_kind):
        old_activity = self.current_activity

        if self.planned_activity is not None:
            self.current_activity = self.planned_activity
            self.planned_activity = None

        elif next_kind == "break":
            self.current_activity = self._make_break_activity(observation)

        else:
            self.current_activity = self._make_role_work_activity(observation)
            if self.current_activity is None:
                self.current_activity = self._make_break_activity(observation)

        old_kind = old_activity.get("kind") if old_activity else "none"
        self._debug_activity(f"AFTER {old_kind}", self.current_activity)

    # ======================================================================
    # Visitor interaction
    # ======================================================================

    def _react_to_visitor(self, observation):
        here = me.position(observation)
        visitor = None

        for person in people.seen(observation):
            if people.is_visitor(person["id"]):
                visitor = person
                break

        if visitor is None:
            return None

        expression = visitor.get("expression", {})

        if expression.get("type") == "wave":
            self.last_was_move = False
            return action.stand(
                geometry.heading_to(here, visitor["position"]),
                "wave",
            )

        current_tick = day.tick(observation)
        if (
            current_tick >= self.next_player_greeting
            and geometry.distance(here, visitor["position"]) <= 3.5
        ):
            self.next_player_greeting = current_tick + self.rng.randint(
                PLAYER_GREETING_MIN_DELAY,
                PLAYER_GREETING_MAX_DELAY,
            )
            self.last_was_move = False
            return action.stand(
                geometry.heading_to(here, visitor["position"]),
                self.rng.choice(("wave", "nod")),
            )

        return None

    # ======================================================================
    # Work execution
    # ======================================================================

    def _perform_work(self, observation, activity):
        prop = activity["prop"]
        purpose = activity["purpose"]

        move = self._follow_activity_route(observation, activity)
        if move is not None:
            return move

        here = me.position(observation)
        prop_position = self._prop_position(observation, prop)
        facing = geometry.heading_to(here, prop_position)

        # While standing at work, prepare the route for the break afterward.
        self._prepare_next_break(observation)

        status = self._work_status(observation, prop, purpose)

        if purpose == "crop" and status == "done" and not self.crop_finished:
            self._debug(
                f"CROP COMPLETE | id={prop['id']} | "
                f"state={self._state_text(observation, prop)}"
            )

        if status == "done":
            if purpose == "crop":
                self.crop_finished = True
            else:
                self.last_role_prop_id = prop["id"]

            self._advance_activity(observation, "break")
            return None

        usable = props.usable(observation)

        if usable is None or usable["id"] != prop["id"]:
            self.last_was_move = True
            return action.walk(facing, SLOW_APPROACH_SPEED, "none")

        self.last_was_move = False

        if purpose == "crop":
            if activity["use_ticks"] == 0:
                self._debug(
                    f"TENDING CROP | id={prop['id']} | "
                    f"state={self._state_text(observation, prop)}"
                )
            activity["use_ticks"] += 1
            return action.stand(facing, "use")

        if purpose == "stall":
            if status == "needs":
                if not activity["used_once"]:
                    activity["used_once"] = True
                    self._debug(
                        f"USING STALL | id={prop['id']} | "
                        f"state={self._state_text(observation, prop)}"
                    )
                    return action.stand(facing, "use")
                return action.stand(facing, "none")

            # Unknown state: do not risk toggling it repeatedly.
            self.last_role_prop_id = prop["id"]
            self._advance_activity(observation, "break")
            return None

        if purpose == "shrine":
            if status == "needs":
                if activity["use_ticks"] == 0:
                    self._debug(
                        f"TENDING SHRINE | id={prop['id']} | "
                        f"state={self._state_text(observation, prop)}"
                    )
                activity["use_ticks"] += 1
                return action.stand(facing, "use")

            # If no useful state is exposed, visibly tend for a short time.
            activity["use_ticks"] += 1
            if activity["use_ticks"] >= GENERIC_WORK_TICKS:
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
            return action.stand(facing, "use")

        if purpose == "repair bench":
            if activity["use_ticks"] == 0:
                self._debug(f"WORKING AT REPAIR BENCH | id={prop['id']}")
            activity["use_ticks"] += 1
            if activity["use_ticks"] >= GENERIC_WORK_TICKS:
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
            return action.stand(facing, "use")

        if purpose == "pump":
            if activity["use_ticks"] == 0:
                self._debug(f"USING PUMP | id={prop['id']}")
            activity["use_ticks"] += 1
            if activity["use_ticks"] >= max(6, GENERIC_WORK_TICKS // 2):
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
            return action.stand(facing, "use")

        if purpose == "board":
            if activity["use_ticks"] == 0:
                self._debug(f"CHECKING BOARD | id={prop['id']}")
            activity["use_ticks"] += 1
            if activity["use_ticks"] >= max(6, GENERIC_WORK_TICKS // 2):
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
            return action.stand(facing, "use")

        if purpose == "bell":
            if not activity["used_once"]:
                activity["used_once"] = True
                self._debug(f"RINGING BELL | id={prop['id']}")
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
                return action.stand(facing, "use")
            return action.stand(facing, "none")

        if purpose == "lantern":
            if status == "done":
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
                return None
            if status == "needs" and not activity["used_once"]:
                activity["used_once"] = True
                self._debug(
                    f"TENDING LANTERN | id={prop['id']} | "
                    f"state={self._state_text(observation, prop)}"
                )
                return action.stand(facing, "use")
            if activity["used_once"]:
                return action.stand(facing, "none")
            self.last_role_prop_id = prop["id"]
            self._advance_activity(observation, "break")
            return None

        if purpose == "hearth":
            if status == "done":
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
                return None

            if activity["use_ticks"] == 0:
                self._debug(
                    f"TENDING HEARTH | id={prop['id']} | "
                    f"state={self._state_text(observation, prop)}"
                )

            # If the hearth exposes a needs/done state, keep tending until
            # it changes. If its state is opaque, cap the cosmetic tending
            # period so the villager can return to normal village life.
            activity["use_ticks"] += 1
            if status == "unknown" and activity["use_ticks"] >= GENERIC_WORK_TICKS:
                self.last_role_prop_id = prop["id"]
                self._advance_activity(observation, "break")
            return action.stand(facing, "use")

        self._advance_activity(observation, "break")
        return None

    # ======================================================================
    # Leisure activities
    # ======================================================================

    def _perform_wander(self, observation, activity):
        if activity.get("target_cell") is None:
            activity["target_cell"] = self._random_outdoor_target(observation)
            activity["route"] = []
        move = self._follow_activity_route(observation, activity)
        if move is not None:
            return move

        self.last_was_move = False
        self._prepare_next_work(observation)

        if day.tick(observation) >= activity["until"]:
            self._advance_activity(observation, "work")
            return None

        return action.stand(me.heading(observation), "none")

    def _perform_sweep(self, observation, activity):
        if activity.get("target_cell") is None:
            activity["target_cell"] = self._random_outdoor_target(observation)
            activity["route"] = []
        move = self._follow_activity_route(observation, activity)
        if move is not None:
            return move

        self.last_was_move = False
        self._prepare_next_work(observation)

        if day.tick(observation) >= activity["until"]:
            self._advance_activity(observation, "work")
            return None

        return action.stand(me.heading(observation), "sweep")

    def _perform_rest(self, observation, activity):
        self.last_was_move = False
        self._prepare_next_work(observation)

        if day.tick(observation) >= activity["until"]:
            self._advance_activity(observation, "work")
            return None

        return action.stand(me.heading(observation), "none")

    def _perform_social(self, observation, activity):
        self.last_was_move = False
        self._prepare_next_work(observation)

        current_tick = day.tick(observation)
        if current_tick >= activity["until"]:
            self._advance_activity(observation, "work")
            return None

        seen = list(people.seen(observation))
        target = None
        target_id = activity.get("target_id")

        if target_id is not None:
            for person in seen:
                if person["id"] == target_id:
                    target = person
                    break

        if target is None:
            nearby_npcs = [
                person
                for person in seen
                if (
                    people.is_npc(person["id"])
                    and person["id"] != self.player_id
                    and geometry.distance(
                        me.position(observation),
                        person["position"],
                    ) <= 3.0
                )
            ]

            if nearby_npcs:
                target = self.rng.choice(nearby_npcs)
                activity["target_id"] = target["id"]
                self._debug(
                    f"CHAT | target={target['id']} | "
                    f"emote={activity['expression']}"
                )

        if target is None:
            return action.stand(me.heading(observation), "none")

        return action.stand(
            geometry.heading_to(
                me.position(observation),
                target["position"],
            ),
            activity["expression"],
        )

    # ======================================================================
    # Leaving home
    # ======================================================================

    def _leave_home(self, observation):
        """Use a small bounded BFS to solve the home's doorway reliably."""
        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)

        if here_cell is None:
            return None

        if layout.ground_at(observation, here_cell) != "interior":
            self.home_route = []
            self.home_stuck_ticks = 0
            self.home_yield_until = 0
            return None

        if not self.home_exit_cells:
            self.last_was_move = False
            return action.stand(me.heading(observation), "none")

        current = _cell_key(here_cell)
        tick = day.tick(observation)

        if self.last_was_move:
            if me.moved(observation) < 0.02:
                self.home_stuck_ticks += 1
            else:
                self.home_stuck_ticks = 0

        if tick < self.home_yield_until:
            self.last_was_move = False
            return action.stand(me.heading(observation), "none")

        # Lower-numbered housemate gets the doorway first.
        nearby_people = list(people.seen(observation)) + list(people.nearby(observation))
        seen_ids = set()
        for person in nearby_people:
            pid = person.get("id")
            if pid in seen_ids or not people.is_npc(pid) or pid == self.player_id:
                continue
            seen_ids.add(pid)
            position = person.get("position")
            if position is None or geometry.distance(here, position) > 1.5:
                continue
            other_cell = layout.cell_at(observation, position)
            if (
                other_cell is not None
                and layout.ground_at(observation, other_cell) == "interior"
                and _seat_number(pid) < self.seat
            ):
                self.home_yield_until = tick + HOME_YIELD_TICKS
                self.last_was_move = False
                return action.stand(me.heading(observation), "none")

        while self.home_route and self.home_route[0] == current:
            self.home_route.pop(0)

        if self.home_route and self.home_route[0] not in self._cached_neighbours(observation, current):
            self.home_route = []

        if self.home_stuck_ticks >= HOME_STUCK_LIMIT:
            self.home_stuck_ticks = 0
            self.home_route = []

        if not self.home_route:
            self.home_route = self._route_with_expansion(
                observation,
                current,
                self.home_exit_cells,
            )

        if not self.home_route:
            self.last_was_move = False
            return action.stand(me.heading(observation), "none")

        target = _cell_centre(observation, _cell_dict(self.home_route[0]))
        self.last_was_move = True
        return action.walk(
            geometry.heading_to(here, target),
            0.55 if len(self.home_route) <= 2 else WALK_SPEED,
            "none",
        )

    # ======================================================================
    # Main behavior tree
    # ======================================================================

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        # 1. Leave home.
        result = self._leave_home(observation)
        if result is not None:
            return result

        # 2. Player interaction can interrupt briefly.
        result = self._react_to_visitor(observation)
        if result is not None:
            return result

        # 3. Ensure there is an activity.
        if self.current_activity is None:
            if self.crop_duty and not self.crop_finished:
                self.current_activity = self._make_work_activity(
                    self.home_crop,
                    "crop",
                    observation,
                )
            else:
                self.current_activity = self._make_role_work_activity(observation)
                if self.current_activity is None:
                    self.current_activity = self._make_break_activity(observation)

            self._debug_activity("NEW", self.current_activity)

        # 4. Continue the current activity.
        kind = self.current_activity["kind"]

        if kind == "work":
            result = self._perform_work(observation, self.current_activity)
        elif kind == "wander":
            result = self._perform_wander(observation, self.current_activity)
        elif kind == "sweep":
            result = self._perform_sweep(observation, self.current_activity)
        elif kind == "rest":
            result = self._perform_rest(observation, self.current_activity)
        elif kind == "social":
            result = self._perform_social(observation, self.current_activity)
        else:
            self.current_activity = None
            result = None

        # If an activity finished this tick, don't recursively start another one.
        if result is None:
            return action.stand(me.heading(observation), "none")

        return result
