import logging
import networkx as nx
import os.path

from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
import unittest

from definitions import ROOT_DIR, OUT_DIR
from src.env.env import NetEnv
from src.network.net import generate_linear_5, Flow, Network


class TestEnvState(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.env = NetEnv()
        self.state_encoder = self.env.state_encoder

    def test_state(self):
        state = self.state_encoder.state()
        logging.debug(state)
        for key, value in state.items():
            logging.debug((key, value.shape))

        logging.debug(self.state_encoder.observation_space)
        self.assertTrue(self.state_encoder.observation_space.contains(state))


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.env = NetEnv()

    def test_check_env(self):
        check_env(self.env)

    def test_action(self):
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))

        route_masks = self.env.action_masks()
        self.assertTrue(route_masks[0])
        obs, reward, done, _, info = self.env.step(0)
        self.assertEqual(obs['decision_phase'][0], 1)

        gating_list = [0, 1, 0]
        for gating in gating_list:
            action_masks = self.env.action_masks()
            self.assertTrue(action_masks[0])
            self.assertTrue(action_masks[1])
            self.assertFalse(action_masks[2])
            obs, reward, done, _, info = self.env.step(gating)
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertFalse(done)

        obs, reward, done, _, info = self.env.step(0)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertFalse(done)

    @unittest.skipIf(os.getenv('RUN_LONG_TESTS') != '1', "Skipping long running test")
    def test_learn_with_action_masks(self):
        model = MaskablePPO("MultiInputPolicy", self.env, verbose=1)
        model.learn(5000)


class TestActionMask(unittest.TestCase):
    def setUp(self):
        self.graph = generate_linear_5()
        self.path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]

    # def test_action_masks_gate(self):
        # flow0 = Flow(f"F0", "E1", "E2", self.path, payload=64, period=2000, jitter=2000)
        # flow0 = Flow(f"F0", "E1", "E2", self.path, payload=64, period=4000, jitter=2000)
        # flows = [flow0]
        # network = Network(self.graph, flows)
        # for link in network.links_dict.values():
        #     link.gcl_capacity = 4


class TestEnvInfo(unittest.TestCase):
    def setUp(self):
        self.graph = generate_linear_5()

        path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]
        flow0 = Flow(f"F0", "E1", "E2", path, payload=64, period=2000, jitter=2000)
        flow1 = Flow(f"F1", "E1", "E2", path, payload=64, period=4000, jitter=2000)
        self.flows = [flow0, flow1]

    def test_success_all_gate(self):
        network = Network(self.graph, self.flows)
        env = NetEnv(network)
        for flow_index in range(2):
            _, _, done, _, info = env.step(0)
            self.assertFalse(done)
            for hop_index in range(3):
                _, _, done, _, info = env.step(1)
                if flow_index == 1 and hop_index == 2:
                    self.assertTrue(done)
                    self.assertTrue(info['success'])
                else:
                    self.assertFalse(done)
                    self.assertFalse(info['success'])

    def test_success_no_gate(self):
        network = Network(self.graph, self.flows)
        env = NetEnv(network)
        for flow_index in range(2):
            _, _, done, _, info = env.step(0)
            self.assertFalse(done)
            for hop_index in range(3):
                _, _, done, _, info = env.step(0)
                if flow_index == 1 and hop_index == 2:
                    self.assertTrue(done)
                    self.assertTrue(info['success'])
                else:
                    self.assertFalse(done)
                    self.assertFalse(info['success'])

    def test_limited_gcl(self):
        network = Network(self.graph, self.flows)
        for link in network.links_dict.values():
            link.gcl_capacity = 5
        env = NetEnv(network)
        env.step(0)
        for _ in range(3):
            action_masks = env.action_masks()
            self.assertTrue(action_masks[1])
            _, _, done, _, _ = env.step(1)
            self.assertFalse(done)

        env.step(0)
        action_masks = env.action_masks()
        self.assertFalse(action_masks[1])
        _, _, done, _, _ = env.step(1)
        self.assertTrue(done)

    def test_action_masks_jitter(self):
        path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]
        flow0 = Flow(f"F0", "E1", "E2", path, payload=64, period=2000, jitter=400)
        flow1 = Flow(f"F1", "E1", "E2", path, payload=64, period=16000, jitter=200)
        flows = [flow0, flow1]
        network = Network(self.graph, flows)
        network.set_gcl(4)
        env = NetEnv(network)

        for flow_index, actions in enumerate(([0, 0, 1], [0, 0, 0])):
            env.step(0)
            for hop_index, action in enumerate(actions):
                action_masks = env.action_masks()
                if flow_index == 1 and hop_index == 2:
                    self.assertFalse(action_masks[0])
                    self.assertFalse(action_masks[1])
                else:
                    self.assertTrue(action_masks[0])
                    self.assertTrue(action_masks[1])

                _, _, done, _, _ = env.step(action)
                if flow_index == 1 and hop_index == 2:
                    self.assertTrue(done)
                else:
                    self.assertFalse(done)


class TestRouteSelection(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([
            ('E1', 'A'), ('A', 'E2'),
            ('E1', 'B'), ('B', 'C'), ('C', 'E2')
        ])
        nx.set_edge_attributes(self.graph, 100, 'link_rate')
        nx.set_node_attributes(self.graph, 'ES', 'node_type')

        shortest_path = [('E1', 'A'), ('A', 'E2')]
        self.flow = Flow('F0', 'E1', 'E2', shortest_path,
                         payload=64, period=2000, jitter=2000)

    def test_select_route_once_then_gate(self):
        env = NetEnv(Network(self.graph, [self.flow]))
        obs, _ = env.reset()

        self.assertEqual(obs['decision_phase'][0], 0)
        self.assertListEqual(env.action_masks().tolist(), [True, True, False])

        obs, reward, done, _, _ = env.step(1)
        self.assertFalse(done)
        self.assertLess(reward, 0)
        self.assertEqual(obs['decision_phase'][0], 1)
        self.assertEqual(self.flow.path, [('E1', 'B'), ('B', 'C'), ('C', 'E2')])
        self.assertFalse(env.action_masks()[2])

    def test_k_one_reduces_to_shortest_route(self):
        env = NetEnv(Network(self.graph, [self.flow]), max_candidate_paths=1)

        self.assertListEqual(env.action_masks().tolist(), [True, False])
        _, reward, done, _, _ = env.step(0)

        self.assertFalse(done)
        self.assertEqual(reward, 0)
        self.assertEqual(self.flow.path, [('E1', 'A'), ('A', 'E2')])
        self.assertListEqual(env.action_masks().tolist(), [True, True])
