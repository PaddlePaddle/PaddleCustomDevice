import paddle
import paddle.base as base
import numpy as np
def sample_output_one_dimension(out, dim):
    # count numbers of different categories
    sample_prob = np.zeros(dim).astype("float32")
    sample_index_prob = np.unique(out, return_counts=True)
    sample_prob[sample_index_prob[0]] = sample_index_prob[1]
    sample_prob /= sample_prob.sum()
    return sample_prob

paddle.enable_static()
paddle.seed(100)
for _ in range(100):
    print(f"start epoch {_}")
    paddle.set_device("mlu:0")
    startup_program = base.Program()
    train_program = base.Program()
    with base.program_guard(train_program, startup_program):
        x = paddle.static.data("x", shape=[4], dtype="float32")
        outs = [paddle.multinomial(x, num_samples=200000, replacement=True) for _ in range(100)]

        out = paddle.concat(outs, axis=0)
        # out = paddle.multinomial(x, num_samples=250000, replacement=True)
        place = base.CustomPlace("mlu", 0)
        exe = base.Executor(place)

    exe.run(startup_program)
    x_np = np.random.rand(4).astype("float32")
    out = exe.run(train_program, feed={"x": x_np}, fetch_list=[out])

    sample_prob = sample_output_one_dimension(out, 4)
    prob = x_np / x_np.sum(axis=-1, keepdims=True)
    print(f"target prob: {prob}")
    np.testing.assert_allclose(sample_prob, prob, rtol=0, atol=0.01)