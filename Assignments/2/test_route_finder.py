import os
import tempfile
import unittest
from math import sqrt

from route_finder import MapState, a_star, h1, sld, read_mars_graph
from Graph import Graph, Node, Edge


class TestMapState(unittest.TestCase):
    def test_eq_and_hash(self):
        state1 = MapState("2,2")
        state2 = MapState("2,2")
        state3 = MapState("3,3")
        self.assertEqual(state1, state2)
        self.assertEqual(hash(state1), hash(state2))
        self.assertNotEqual(state1, state3)

    def test_repr(self):
        state = MapState("5,5")
        self.assertEqual(repr(state), "(5,5)")

    def test_ordering(self):
        state1 = MapState("2,2", g=1, h=2)
        state2 = MapState("3,3", g=1, h=3)
        self.assertTrue(state1 < state2)
        self.assertTrue(state1 <= state2)
        self.assertFalse(state2 < state1)
        self.assertFalse(state2 <= state1)

    def test_is_goal(self):
        goal_state = MapState("1,1")
        non_goal_state = MapState("2,2")
        self.assertTrue(goal_state.is_goal())
        self.assertFalse(non_goal_state.is_goal())

    def test_successors(self):
        graph = Graph()
        node22 = Node("2,2")
        node23 = Node("2,3")
        graph.add_node(node22)
        graph.add_node(node23)
        edge = Edge(node22, node23, 1)
        graph.add_edge(edge)
        state = MapState("2,2", mars_graph=graph)
        successors = state.successors()
        self.assertEqual(len(successors), 1)
        self.assertEqual(successors[0].location, "2,3")
        self.assertEqual(successors[0].g, state.g + 1)
        self.assertIs(successors[0].prev_state, state)


class TestHeuristics(unittest.TestCase):
    def test_h1(self):
        state = MapState("3,3")
        self.assertEqual(h1(state), 0)

    def test_sld(self):
        state = MapState("4,4")
        expected = sqrt((4 - 1) ** 2 + (4 - 1) ** 2)
        self.assertAlmostEqual(sld(state), expected, places=5)


class TestAStar(unittest.TestCase):
    def setUp(self):
        # Acyclic graph for a successful path: "2,2" -> "1,2" -> "1,1"
        self.acyclic_graph = Graph()
        self.n22 = Node("2,2")
        self.n12 = Node("1,2")
        self.n11 = Node("1,1")
        for n in (self.n22, self.n12, self.n11):
            self.acyclic_graph.add_node(n)
        self.acyclic_graph.add_edge(Edge(self.n22, self.n12, 1))
        self.acyclic_graph.add_edge(Edge(self.n12, self.n11, 1))

        # Graph with no solution: single node not goal.
        self.no_solution_graph = Graph()
        self.n55 = Node("5,5")
        self.no_solution_graph.add_node(self.n55)

        # Cyclic graph:
        # "2,2" -> "2,3", "2,3" -> "2,2", and "2,3" -> "1,1"
        self.cyclic_graph = Graph()
        self.c_n22 = Node("2,2")
        self.c_n23 = Node("2,3")
        self.c_n11 = Node("1,1")
        for n in (self.c_n22, self.c_n23, self.c_n11):
            self.cyclic_graph.add_node(n)
        self.cyclic_graph.add_edge(Edge(self.c_n22, self.c_n23, 1))
        self.cyclic_graph.add_edge(Edge(self.c_n23, self.c_n22, 1))
        self.cyclic_graph.add_edge(Edge(self.c_n23, self.c_n11, 1))

    def test_a_star_success(self):
        start_state = MapState("2,2", mars_graph=self.acyclic_graph)
        goal_state = a_star(start_state, sld, MapState.is_goal)
        self.assertIsNotNone(goal_state)
        self.assertEqual(goal_state.location, "1,1")
        path = []
        current = goal_state
        while current:
            path.append(current.location)
            current = current.prev_state
        self.assertEqual(path[::-1], ["2,2", "1,2", "1,1"])

    def test_a_star_no_solution(self):
        start_state = MapState("5,5", mars_graph=self.no_solution_graph)
        goal_state = a_star(start_state, sld, MapState.is_goal)
        self.assertIsNone(goal_state)

    def test_a_star_with_cycle(self):
        start_state = MapState("2,2", mars_graph=self.cyclic_graph)
        goal_state = a_star(start_state, sld, MapState.is_goal, use_closed_list=True)
        self.assertIsNotNone(goal_state)
        self.assertEqual(goal_state.location, "1,1")

    def test_a_star_no_closed_list(self):
        start_state = MapState("2,2", mars_graph=self.acyclic_graph)
        goal_state = a_star(start_state, sld, MapState.is_goal, use_closed_list=False)
        self.assertIsNotNone(goal_state)
        self.assertEqual(goal_state.location, "1,1")
        path = []
        current = goal_state
        while current:
            path.append(current.location)
            current = current.prev_state
        self.assertEqual(path[::-1], ["2,2", "1,2", "1,1"])

    def test_uniform_cost_success(self):
        start_state = MapState("2,2", mars_graph=self.acyclic_graph)
        goal_state = a_star(start_state, h1, MapState.is_goal)
        self.assertIsNotNone(goal_state)
        self.assertEqual(goal_state.location, "1,1")
        path = []
        current = goal_state
        while current:
            path.append(current.location)
            current = current.prev_state
        self.assertEqual(path[::-1], ["2,2", "1,2", "1,1"])

    def test_uniform_cost_no_solution(self):
        start_state = MapState("5,5", mars_graph=self.no_solution_graph)
        goal_state = a_star(start_state, h1, MapState.is_goal)
        self.assertIsNone(goal_state)

    def test_uniform_cost_with_cycle(self):
        start_state = MapState("2,2", mars_graph=self.cyclic_graph)
        goal_state = a_star(start_state, h1, MapState.is_goal, use_closed_list=True)
        self.assertIsNotNone(goal_state)
        self.assertEqual(goal_state.location, "1,1")

    def test_uniform_cost_no_closed_list(self):
        start_state = MapState("2,2", mars_graph=self.acyclic_graph)
        goal_state = a_star(start_state, h1, MapState.is_goal, use_closed_list=False)
        self.assertIsNotNone(goal_state)
        self.assertEqual(goal_state.location, "1,1")
        path = []
        current = goal_state
        while current:
            path.append(current.location)
            current = current.prev_state
        self.assertEqual(path[::-1], ["2,2", "1,2", "1,1"])


if __name__ == "__main__":
    unittest.main()
