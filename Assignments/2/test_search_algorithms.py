import unittest
from mars_planner import RoverState, action_list, mission_complete, battery_goal
from search_algorithms import (
    breadth_first_search,
    depth_first_search,
    iterative_deepening_search,
)


class TestBFS(unittest.TestCase):
    def test_bfs_battery_goal(self):
        result = breadth_first_search(RoverState(), action_list, battery_goal)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")

    def test_bfs_station_sample_goal(self):
        def goal(s):
            return s.rover_loc == "station" and s.sample_loc == "station"

        result = breadth_first_search(RoverState(), action_list, goal)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "station")
        self.assertEqual(result[0].sample_loc, "station")

    def test_bfs_mission_complete(self):
        result = breadth_first_search(RoverState(), action_list, mission_complete)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")
        self.assertTrue(result[0].charged)
        self.assertEqual(result[0].sample_loc, "station")

    def test_bfs_no_goal(self):
        def goal(s):
            return s.rover_loc == "moon"  # Unreachable goal

        result = breadth_first_search(RoverState(), action_list, goal)
        self.assertIsNone(result)

    def test_bfs_no_closed_list(self):
        result = breadth_first_search(
            RoverState(), action_list, battery_goal, use_closed_list=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")


class TestDFS(unittest.TestCase):
    def test_dfs_battery_goal(self):
        result = depth_first_search(RoverState(), action_list, battery_goal)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")

    def test_dfs_station_sample_goal(self):
        def goal(s):
            return s.rover_loc == "station" and s.sample_loc == "station"

        result = depth_first_search(RoverState(), action_list, goal)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "station")
        self.assertEqual(result[0].sample_loc, "station")

    def test_dfs_mission_complete(self):
        result = depth_first_search(RoverState(), action_list, mission_complete)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")
        self.assertTrue(result[0].charged)
        self.assertEqual(result[0].sample_loc, "station")

    def test_dfs_no_goal(self):
        def goal(s):
            return s.rover_loc == "moon"

        result = depth_first_search(RoverState(), action_list, goal)

        self.assertIsNone(result)

    def test_dfs_no_closed_list(self):
        result = depth_first_search(
            RoverState(), action_list, battery_goal, use_closed_list=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")


class TestDLS(unittest.TestCase):
    def test_dls_battery_goal(self):
        result = depth_first_search(RoverState(), action_list, battery_goal, limit=3)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")

    def test_dls_station_sample_goal(self):
        def goal(s):
            return s.rover_loc == "station" and s.sample_loc == "station"

        result = depth_first_search(RoverState(), action_list, goal, limit=4)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "station")
        self.assertEqual(result[0].sample_loc, "station")

    def test_dls_mission_complete(self):
        result = depth_first_search(
            RoverState(), action_list, mission_complete, limit=7
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")
        self.assertTrue(result[0].charged)
        self.assertEqual(result[0].sample_loc, "station")

    def test_dls_no_goal(self):
        def goal(s):
            return s.rover_loc == "moon"

        result = depth_first_search(RoverState(), action_list, goal, limit=10)
        self.assertIsNone(result)

    def test_dls_low_limit(self):
        # With a very low depth limit, the mission_complete goal should be unreachable.
        def goal(s):
            return mission_complete(s)

        result = depth_first_search(RoverState(), action_list, goal, limit=1)
        self.assertIsNone(result)

    def test_dls_no_closed_list(self):
        result = depth_first_search(
            RoverState(), action_list, battery_goal, use_closed_list=False, limit=10
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")


class TestIDS(unittest.TestCase):
    def test_ids_battery_goal(self):
        result = iterative_deepening_search(RoverState(), action_list, battery_goal)

        self.assertIsNotNone(result)
        # If the DFS limit is not provided, our DFS returns (state, action)
        self.assertEqual(result[0].rover_loc, "battery")

    def test_ids_station_sample_goal(self):
        def goal(s):
            return s.rover_loc == "station" and s.sample_loc == "station"

        result = iterative_deepening_search(RoverState(), action_list, goal)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "station")
        self.assertEqual(result[0].sample_loc, "station")

    def test_ids_mission_complete(self):
        result = iterative_deepening_search(RoverState(), action_list, mission_complete)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")
        self.assertTrue(result[0].charged)
        self.assertEqual(result[0].sample_loc, "station")

    def test_ids_no_closed_list(self):
        result = iterative_deepening_search(
            RoverState(), action_list, battery_goal, use_closed_list=False
        )

        self.assertIsNotNone(result)
        self.assertEqual(result[0].rover_loc, "battery")


if __name__ == "__main__":
    unittest.main()
