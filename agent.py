"""Three Branches village-life agent.

Priorities:
1. Leave home.
2. Wave back when the visitor waves.
3. Tend assigned shrines and stalls.
4. Wander naturally around the village.
5. Use cell-based pathfinding to avoid walls, scenery, and props.
"""

from collections import deque

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, geometry, layout, me, people, props


# How long a villager visibly tends one workplace.
TEND_TICKS = 24

# Movement below this for several attempted walking ticks means we may be stuck.
STUCK_LIMIT = 5

# Slower than maximum makes the villagers look a little less frantic
# and reduces overshooting around corners.
WALK_SPEED = 0.8


def _cell_key(cell: dict[str, int]) -> tuple[int, int]:
    return cell["x"], cell["y"]


def _cell_dict(key: tuple[int, int]) -> dict[str, int]:
    return {"x": key[0], "y": key[1]}


def _cell_centre(
    observation: ThreeBranchesObservation,
    cell: dict[str, int],
) -> dict[str, float]:
    """Return the world-space centre of a village grid cell."""

    frame = layout.frame(observation)
    size = frame.get("cell_size", 1.0)

    return {
        "x": (cell["x"] + 0.5) * size,
        "y": (cell["y"] + 0.5) * size,
    }


def _is_work_prop(prop: dict) -> bool:
    """Recognize shrine/stall variants without depending on exact type names."""

    prop_type = str(prop.get("type", "")).lower()

    return (
        "shrine" in prop_type
        or "stall" in prop_type
    )


class Agent:
    """Villager that works first, then wanders and socializes."""

    def reset(
        self,
        seed: int,
        observation: ThreeBranchesObservation,
    ) -> None:
        """Build static navigation and choose this villager's jobs."""

        self.seed = seed
        self.rng = me.rng(observation, seed)

        # --------------------------------------------------------------
        # Navigation graph
        # --------------------------------------------------------------

        frame = layout.frame(observation)

        self.graph: dict[
            tuple[int, int],
            list[tuple[int, int]]
        ] = {}

        self.outdoor_cells: list[tuple[int, int]] = []

        for x in range(frame["cells_x"]):
            for y in range(frame["cells_y"]):
                cell = {"x": x, "y": y}

                if not layout.walkable(observation, cell):
                    continue

                key = (x, y)
                self.graph[key] = []

                # Wander only on outdoor ground.
                if layout.ground_at(observation, cell) != "interior":
                    self.outdoor_cells.append(key)

        # Build only legal cardinal connections.
        directions = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        )

        for key in self.graph:
            start = _cell_dict(key)

            for dx, dy in directions:
                neighbour_key = (
                    key[0] + dx,
                    key[1] + dy,
                )

                if neighbour_key not in self.graph:
                    continue

                neighbour = _cell_dict(neighbour_key)

                if layout.can_step(
                    observation,
                    start,
                    neighbour,
                ):
                    self.graph[key].append(neighbour_key)

        # --------------------------------------------------------------
        # Work assignment
        # --------------------------------------------------------------

        all_work = [
            prop
            for prop in props.all(observation)
            if _is_work_prop(prop)
        ]

        all_work.sort(key=lambda p: p["id"])

        # Find our stable position in the NPC roster.
        my_id = me.player_id(observation)

        npc_ids = [
            person["id"]
            for person in people.roster(observation)
            if people.is_npc(person["id"])
        ]

        try:
            my_slot = npc_ids.index(my_id)
        except ValueError:
            my_slot = 0

        npc_count = max(1, len(npc_ids))

        # Primarily partition jobs between villagers so everyone does not
        # rush to the same shrine/stall.
        self.work_queue = [
            prop
            for index, prop in enumerate(all_work)
            if index % npc_count == my_slot
        ]

        # If there are fewer workplaces than villagers, give idle villagers
        # one rotating workplace instead of giving them no job whatsoever.
        if not self.work_queue and all_work:
            self.work_queue = [
                all_work[my_slot % len(all_work)]
            ]

        self.completed_work: set[str] = set()
        self.current_work = None
        self.tend_ticks = 0

        # --------------------------------------------------------------
        # Movement state
        # --------------------------------------------------------------

        self.route: list[tuple[int, int]] = []
        self.route_goal = None

        self.wander_target = None

        self.last_was_move = False
        self.stuck_ticks = 0

    # ------------------------------------------------------------------
    # Social behavior
    # ------------------------------------------------------------------

    def _waving_visitor(self, observation):
        """Return the visitor if they are visible and currently waving."""

        for person in people.seen(observation):
            if not people.is_visitor(person["id"]):
                continue

            expression = person.get("expression", {})

            if expression.get("type") == "wave":
                return person

        return None

    # ------------------------------------------------------------------
    # Work selection
    # ------------------------------------------------------------------

    def _next_work(self):
        """Return this villager's next unfinished workplace."""

        if (
            self.current_work is not None
            and self.current_work["id"] not in self.completed_work
        ):
            return self.current_work

        self.current_work = None
        self.tend_ticks = 0

        for prop in self.work_queue:
            if prop["id"] not in self.completed_work:
                self.current_work = prop
                return prop

        return None

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _approach_cells(
        self,
        prop: dict,
    ) -> list[tuple[int, int]]:
        """Find walkable cells adjacent to a prop."""

        cell = prop["cell"]
        x = cell["x"]
        y = cell["y"]

        candidates = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )

        return [
            candidate
            for candidate in candidates
            if candidate in self.graph
        ]

    def _find_route(
        self,
        observation: ThreeBranchesObservation,
        goals: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Breadth-first route from our current cell to any goal."""

        here = layout.cell_at(
            observation,
            me.position(observation),
        )

        if here is None:
            return []

        start = _cell_key(here)

        if start in goals:
            return []

        if start not in self.graph:
            return []

        goal_set = set(goals)

        queue = deque([start])
        previous = {start: None}

        found = None

        while queue:
            current = queue.popleft()

            if current in goal_set:
                found = current
                break

            for neighbour in self.graph.get(current, []):
                if neighbour in previous:
                    continue

                previous[neighbour] = current
                queue.append(neighbour)

        if found is None:
            return []

        path = []

        current = found

        while current != start:
            path.append(current)
            current = previous[current]

        path.reverse()

        return path

    def _move_along_route(
        self,
        observation: ThreeBranchesObservation,
        goals: list[tuple[int, int]],
        goal_id,
        expression: str = "none",
    ) -> ThreeBranchesAction:
        """Follow or rebuild a safe cell route."""

        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)

        if here_cell is None:
            self.last_was_move = False
            return action.stand(
                me.heading(observation),
                expression,
            )

        current_key = _cell_key(here_cell)

        # --------------------------------------------------------------
        # Detect failed movement
        # --------------------------------------------------------------

        if self.last_was_move:
            if me.moved(observation) < 0.02:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0

        # Force a new route if another villager or some temporary condition
        # has kept us motionless for several ticks.
        if self.stuck_ticks >= STUCK_LIMIT:
            self.route = []
            self.route_goal = None
            self.stuck_ticks = 0

            # Take a legal side-step if possible.
            neighbours = list(
                self.graph.get(current_key, [])
            )

            if neighbours:
                sidestep = self.rng.choice(neighbours)

                target = _cell_centre(
                    observation,
                    _cell_dict(sidestep),
                )

                self.last_was_move = True

                return action.walk(
                    geometry.heading_to(here, target),
                    WALK_SPEED,
                    expression,
                )

        # --------------------------------------------------------------
        # Rebuild route if destination changed
        # --------------------------------------------------------------

        if self.route_goal != goal_id:
            self.route = []
            self.route_goal = goal_id

        if not self.route:
            self.route = self._find_route(
                observation,
                goals,
            )

        # Already standing at one of the target cells.
        if current_key in goals:
            self.last_was_move = False
            return action.stand(
                me.heading(observation),
                expression,
            )

        if not self.route:
            self.last_was_move = False
            return action.stand(
                me.heading(observation),
                expression,
            )

        # Drop any route nodes we already reached.
        while self.route and self.route[0] == current_key:
            self.route.pop(0)

        if not self.route:
            self.last_was_move = False
            return action.stand(
                me.heading(observation),
                expression,
            )

        next_cell = _cell_dict(self.route[0])

        target = _cell_centre(
            observation,
            next_cell,
        )

        self.last_was_move = True

        return action.walk(
            geometry.heading_to(here, target),
            WALK_SPEED,
            expression,
        )

    # ------------------------------------------------------------------
    # Work behavior
    # ------------------------------------------------------------------

    def _tend_workplace(
        self,
        observation: ThreeBranchesObservation,
        work: dict,
    ) -> ThreeBranchesAction:
        """Travel to and visibly tend a shrine or stall."""

        here = me.position(observation)
        work_position = _cell_centre(
            observation,
            work["cell"],
        )

        approach_cells = self._approach_cells(work)

        if not approach_cells:
            # Cannot reach this prop safely.
            self.completed_work.add(work["id"])
            self.current_work = None
            self.route = []
            self.route_goal = None

            self.last_was_move = False

            return action.stand(
                me.heading(observation),
                "none",
            )

        here_cell = layout.cell_at(observation, here)

        if (
            here_cell is not None
            and _cell_key(here_cell) in approach_cells
        ):
            self.last_was_move = False

            facing = geometry.heading_to(
                here,
                work_position,
            )

            usable = props.usable(observation)

            # If the engine confirms this exact prop can be used,
            # use it. We never blindly "use", because another nearby
            # prop could otherwise be selected.
            if (
                usable is not None
                and usable["id"] == work["id"]
            ):
                self.tend_ticks += 1

                if self.tend_ticks >= TEND_TICKS:
                    self.completed_work.add(work["id"])
                    self.current_work = None
                    self.tend_ticks = 0
                    self.route = []
                    self.route_goal = None

                return action.stand(
                    facing,
                    "use",
                )

            # A stall/shrine may not expose a usable interaction from
            # precisely this position. "Sweep" still gives the visitor
            # a clear visual that the villager is tending the location.
            self.tend_ticks += 1

            if self.tend_ticks >= TEND_TICKS:
                self.completed_work.add(work["id"])
                self.current_work = None
                self.tend_ticks = 0
                self.route = []
                self.route_goal = None

            return action.stand(
                facing,
                "sweep",
            )

        return self._move_along_route(
            observation,
            approach_cells,
            ("work", work["id"]),
        )

    # ------------------------------------------------------------------
    # Wandering
    # ------------------------------------------------------------------

    def _wander(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction:
        """Pick reachable outdoor destinations and stroll between them."""

        here_cell = layout.cell_at(
            observation,
            me.position(observation),
        )

        if here_cell is None or not self.outdoor_cells:
            self.last_was_move = False

            return action.stand(
                me.heading(observation),
                "none",
            )

        current = _cell_key(here_cell)

        # Select a new destination if there isn't one, or when we arrive.
        if (
            self.wander_target is None
            or self.wander_target == current
        ):
            choices = [
                cell
                for cell in self.outdoor_cells
                if cell != current
            ]

            if not choices:
                self.last_was_move = False
                return action.stand(
                    me.heading(observation),
                    "none",
                )

            self.wander_target = self.rng.choice(choices)
            self.route = []
            self.route_goal = None

        result = self._move_along_route(
            observation,
            [self.wander_target],
            ("wander", self.wander_target),
        )

        return result

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def act(
        self,
        observation: ThreeBranchesObservation,
    ) -> ThreeBranchesAction:
        """Choose this tick's village behavior."""

        here = me.position(observation)
        heading = me.heading(observation)

        # --------------------------------------------------------------
        # 1. Social reaction: wave back at the visitor.
        # --------------------------------------------------------------

        visitor = self._waving_visitor(observation)

        if visitor is not None:
            self.last_was_move = False

            return action.stand(
                geometry.heading_to(
                    here,
                    visitor["position"],
                ),
                "wave",
            )

        # --------------------------------------------------------------
        # 2. Leave home before beginning the day's work.
        # --------------------------------------------------------------

        home = me.home(observation)
        here_cell = layout.cell_at(observation, here)

        if (
            home != "none"
            and here_cell is not None
            and layout.ground_at(
                observation,
                here_cell,
            ) == "interior"
        ):
            door = layout.doorway(
                observation,
                home,
            )

            if door is not None:
                self.last_was_move = True

                return action.walk(
                    geometry.heading_to(
                        here,
                        door,
                    ),
                    WALK_SPEED,
                    "none",
                )

        # --------------------------------------------------------------
        # 3. Work before leisure.
        # --------------------------------------------------------------

        work = self._next_work()

        if work is not None:
            return self._tend_workplace(
                observation,
                work,
            )

        # --------------------------------------------------------------
        # 4. Once work is done, wander through the village.
        # --------------------------------------------------------------

        return self._wander(observation)