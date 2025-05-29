import unittest
from mars_planner import *


class TestRoverState(unittest.TestCase):
    def test_init(self):
        s = RoverState()
        self.assertEqual(s.rover_loc, "station")
        self.assertEqual(s.sample_loc, "site")
        self.assertFalse(s.holding_sample)
        self.assertFalse(s.charged)
        self.assertFalse(s.holding_tool)
        self.assertIsNone(s.prev)
        self.assertEqual(s.depth, 0)

        s2 = RoverState("battery", "station", True, True, True)
        self.assertEqual(s2.rover_loc, "battery")
        self.assertEqual(s2.sample_loc, "station")
        self.assertTrue(s2.holding_sample)
        self.assertTrue(s2.charged)
        self.assertTrue(s2.holding_tool)
        self.assertIsNone(s2.prev)
        self.assertEqual(s.depth, 0)

    def test_eq(self):
        s1 = RoverState("station", "site", False, False, False)
        s2 = RoverState("station", "site", False, False, False)
        self.assertEqual(s1, s2)

        s3 = RoverState("sample", "site", False, False, False)
        self.assertNotEqual(s1, s3)

        s4 = RoverState("station", "site", True, False, False)
        self.assertNotEqual(s1, s4)

        s5 = RoverState("station", "site", False, True, False)
        self.assertNotEqual(s1, s5)

        s6 = RoverState("station", "site", False, False, True)
        self.assertNotEqual(s1, s6)

        s7 = RoverState("station", "site", False, False, False, s6)
        self.assertEqual(s1, s7)

        s8 = RoverState("station", "site", False, False, False)
        s8.depth = 1
        self.assertEqual(s1, s8)

    def test_hash(self):
        s1 = RoverState("station", "site", False, False, False)
        s2 = RoverState("station", "site", False, False, False)
        self.assertEqual(hash(s1), hash(s2))
        states_set = {s1, s2}
        self.assertEqual(len(states_set), 1)

    def test_repr(self):
        s = RoverState("station", "site", True, True, True)
        expected = "Rover Location: station\nSample Location: site\nHolding Sample?: True\nCharged? True\nHolding Tool? True"
        self.assertEqual(repr(s), expected)

    # --- Tests for move_to_sample ---
    def test_move_to_site_without_holding(self):
        state = RoverState(sample_loc="station")
        new_state = move_to_site(state)
        self.assertEqual(new_state.rover_loc, "site")
        self.assertEqual(new_state.sample_loc, "station")
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    def test_move_to_site_with_holding(self):
        state = RoverState(sample_loc="station", holding_sample=True)
        new_state = move_to_site(state)
        self.assertEqual(new_state.rover_loc, "site")
        self.assertEqual(new_state.sample_loc, "site")
        self.assertTrue(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    # --- Tests for move_to_station ---
    def test_move_to_station_without_holding(self):
        state = RoverState("site")
        new_state = move_to_station(state)
        self.assertEqual(new_state.rover_loc, "station")
        self.assertEqual(new_state.sample_loc, "site")
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    def test_move_to_station_with_holding(self):
        state = RoverState("site", holding_sample=True)
        new_state = move_to_station(state)
        self.assertEqual(new_state.rover_loc, "station")
        self.assertEqual(new_state.sample_loc, "station")
        self.assertTrue(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    # --- Tests for move_to_battery ---
    def test_move_to_battery_without_holding(self):
        state = RoverState(sample_loc="station")
        new_state = move_to_battery(state)
        self.assertEqual(new_state.rover_loc, "battery")
        self.assertEqual(new_state.sample_loc, "station")
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    def test_move_to_battery_with_holding(self):
        state = RoverState(sample_loc="station", holding_sample=True)
        new_state = move_to_battery(state)
        self.assertEqual(new_state.rover_loc, "battery")
        self.assertEqual(new_state.sample_loc, "battery")
        self.assertTrue(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    # --- Tests for pick_up_sample ---
    def test_pick_up_sample_success(self):
        # The rover is at the same location as the sample.
        state = RoverState("site")
        new_state = pick_up_sample(state)
        self.assertTrue(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    def test_pick_up_sample_failure(self):
        # The rover is not at the sample location so nothing should happen.
        state = RoverState()
        new_state = pick_up_sample(state)
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)
        # Also, if already holding the sample, nothing changes.
        state_already = RoverState("site", holding_sample=True)
        new_state_already = pick_up_sample(state_already)
        self.assertTrue(new_state_already.holding_sample)
        self.assertEqual(new_state_already.prev, state_already)
        self.assertEqual(new_state_already, state_already)

    # --- Tests for drop_sample ---
    def test_drop_sample_success(self):
        # The rover is at the same location as the sample.
        state = RoverState("site", holding_sample=True)
        new_state = drop_sample(state)
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    def test_drop_sample_failure(self):
        # The rover is not at the sample location so no change occurs.
        state = RoverState()
        new_state = drop_sample(state)
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)
        self.assertEqual(new_state, state)  # no effective change

    # --- Tests for pick up tool ---
    def test_pick_up_tool_when_not_holding(self):
        state = RoverState()
        new_state = pick_up_tool(state)
        self.assertTrue(new_state.holding_tool)
        self.assertEqual(new_state.prev, state)

    def test_pick_up_tool_when_already_holding(self):
        state = RoverState(holding_tool=True)
        new_state = pick_up_tool(state)
        self.assertTrue(new_state.holding_tool)
        self.assertEqual(new_state.prev, state)
        self.assertEqual(new_state, state)

    # --- Tests for drop tool ---
    def test_drop_tool_when_holding(self):
        state = RoverState(holding_tool=True)
        new_state = drop_tool(state)
        self.assertFalse(new_state.holding_tool)
        self.assertEqual(new_state.prev, state)

    def test_drop_tool_when_not_holding(self):
        state = RoverState()
        new_state = drop_tool(state)
        self.assertFalse(new_state.holding_tool)
        self.assertEqual(new_state.prev, state)
        self.assertEqual(new_state, state)

    # --- Tests for use tool ---
    def test_use_tool_success(self):
        state = RoverState("site", holding_tool=True)
        new_state = use_tool(state)
        self.assertTrue(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)

    def test_use_tool_failure_no_tool(self):
        state = RoverState("site")
        new_state = use_tool(state)
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)
        self.assertEqual(new_state, state)

    def test_use_tool_failure_wrong_location(self):
        state = RoverState(holding_tool=True)
        new_state = use_tool(state)
        self.assertFalse(new_state.holding_sample)
        self.assertEqual(new_state.prev, state)
        self.assertEqual(new_state, state)

    # --- Tests for charge ---
    def test_charge_success(self):
        # The rover must be at "battery" for a successful charge.
        state = RoverState("battery")
        new_state = charge(state)
        self.assertTrue(new_state.charged)
        self.assertEqual(new_state.prev, state)

    def test_charge_failure(self):
        state = RoverState()
        new_state = charge(state)
        self.assertFalse(new_state.charged)
        self.assertEqual(new_state.prev, state)
        self.assertEqual(new_state, state)  # no effective change

    # --- Tests for the successors method ---
    def test_successors_filters_no_change(self):
        # In a state where some actions have no effect, they should be filtered out.
        state = RoverState()
        # For example, move_to_station should produce no effective change when already at "station".
        actions = [move_to_station, move_to_site, pick_up_sample]
        successors = state.successors(actions)
        # Ensure that none of the successors equal the original state.
        for new_state, action_name in successors:
            self.assertNotEqual(new_state, state)
        # Verify that move_to_station is not among the returned actions.
        action_names = [name for (_, name) in successors]
        self.assertNotIn("move_to_station", action_names)
        # move_to_sample should cause a change.
        self.assertIn("move_to_site", action_names)

    def test_successors_all_actions_change(self):
        # Create a state where every action produces a change.
        state = RoverState("unknown", "unknown")
        actions = [move_to_station, move_to_site, move_to_battery]
        successors = state.successors(actions)
        self.assertEqual(len(successors), 3)
        expected_locations = {"station", "site", "battery"}
        actual_locations = {new_state.rover_loc for new_state, _ in successors}
        self.assertEqual(actual_locations, expected_locations)

    # --- Tests for prev_chain ---
    def test_prev_chain(self):
        state1 = RoverState()
        state2 = move_to_site(state1)
        state3 = pick_up_sample(state2)
        self.assertEqual(state3.prev, state2)
        self.assertEqual(state2.prev, state1)

    # --- Tests for goal functions ---
    def test_battery_goal(self):
        state = RoverState(
            "battery",
        )
        self.assertTrue(battery_goal(state))
        state = RoverState()
        self.assertFalse(battery_goal(state))

    def test_charged_goal(self):
        state = RoverState(charged=True)
        self.assertTrue(charged_goal(state))
        state = RoverState()
        self.assertFalse(charged_goal(state))

    def test_sample_goal(self):
        state = RoverState(sample_loc="station")
        self.assertTrue(sample_goal(state))
        state = RoverState()
        self.assertFalse(sample_goal(state))

    def test_mission_complete(self):
        # mission_complete is True only if all three goals are met.
        state = RoverState("battery", "station", charged=True)
        self.assertTrue(mission_complete(state))
        state = RoverState("battery", "station")
        self.assertFalse(mission_complete(state))
        state = RoverState(sample_loc="station", charged=True)
        self.assertFalse(mission_complete(state))
        state = RoverState("battery", charged=True)
        self.assertFalse(mission_complete(state))
