import torch


def generate_export_tests():
    tests = []

    def extract_nodes(ep):
        nodes = []
        for node in ep.graph.nodes:
            nd = {
                "op": node.op,
                "name": node.name,
                "target": str(node.target),
            }
            nd["args"] = []
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    nd["args"].append(a.name)
                elif isinstance(a, tuple):
                    nd["args"].append([n.name if isinstance(n, torch.fx.Node) else str(n) for n in a])
                else:
                    nd["args"].append(str(a))
            if "val" in node.meta:
                v = node.meta["val"]
                if isinstance(v, torch.Tensor):
                    nd["val_shape"] = list(v.shape)
            nodes.append(nd)
        return nodes

    def extract_specs(ep):
        sig = ep.graph_signature
        input_specs = [
            {"kind": s.kind.name, "name": s.arg.name if hasattr(s.arg, "name") else str(s.arg)} for s in sig.input_specs
        ]
        output_specs = [
            {"kind": s.kind.name, "name": s.arg.name if hasattr(s.arg, "name") else str(s.arg)}
            for s in sig.output_specs
        ]
        return input_specs, output_specs

    # Test 1: Simple Sequential(Linear, ReLU)
    torch.manual_seed(42)
    model1 = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
    x1 = torch.randn(2, 3)
    ep1 = torch.export.export(model1, (x1,))

    nodes1 = extract_nodes(ep1)
    input_specs1, output_specs1 = extract_specs(ep1)

    tests.append(
        {
            "test_name": "linear_relu",
            "model_type": "LinearReLU",
            "input": x1.detach().numpy().tolist(),
            "weight": model1[0].weight.detach().numpy().tolist(),
            "bias": model1[0].bias.detach().numpy().tolist(),
            "in_features": 3,
            "out_features": 2,
            "expected_nodes": nodes1,
            "expected_input_specs": input_specs1,
            "expected_output_specs": output_specs1,
        }
    )

    # Test 2: Sequential(Linear, ReLU, Linear, Sigmoid)
    torch.manual_seed(42)
    model2 = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2), torch.nn.Sigmoid())
    x2 = torch.randn(2, 4)
    ep2 = torch.export.export(model2, (x2,))

    nodes2 = extract_nodes(ep2)
    input_specs2, output_specs2 = extract_specs(ep2)

    tests.append(
        {
            "test_name": "two_layer",
            "model_type": "TwoLayer",
            "input": x2.detach().numpy().tolist(),
            "linear1_weight": model2[0].weight.detach().numpy().tolist(),
            "linear1_bias": model2[0].bias.detach().numpy().tolist(),
            "linear2_weight": model2[2].weight.detach().numpy().tolist(),
            "linear2_bias": model2[2].bias.detach().numpy().tolist(),
            "linear1_in": 4,
            "linear1_out": 3,
            "linear2_in": 3,
            "linear2_out": 2,
            "expected_nodes": nodes2,
            "expected_input_specs": input_specs2,
            "expected_output_specs": output_specs2,
        }
    )

    return tests
