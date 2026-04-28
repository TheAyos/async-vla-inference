# Understanding Asynchronous Inference Methods for Vision-Language-Action Models

This repository accompanies Ayoub Agouzoul's bachelor's thesis, completed as part of his Bachelor of Science degree at École polytechnique. It presents a systematic, head-to-head comparison of four recently proposed methods for handling the inference-latency / observation-staleness gap that appears when Vision-Language-Action (VLA) policies execute inference asynchronously:

- **IT-RTC** — Inference-Time Real-Time Chunking ([Black et al., 2025](https://arxiv.org/abs/2506.07339))
- **TT-RTC** — Training-Time Real-Time Chunking ([Black et al., 2025](https://arxiv.org/abs/2512.05964))
- **VLASH** — Visual-Lag, Actual-State Hybrid ([Tang et al., 2025](https://arxiv.org/abs/2512.01031))
- **A2C2** — Additive Action-Chunk Correction ([Saito et al., 2025](https://arxiv.org/abs/2509.23224))

Each method has, until the start of this work, been published within a different codebase, base policy, and evaluation protocol. The two unified codebases hosted here (one per benchmark) integrate all four methods on a shared backbone, a shared dataset format, and shared evaluation code so that a fair comparison is possible.

## Repository contents

| Path                                   | Description                                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------------- |
| [`bt-kinetix/`](bt-kinetix/)           | Kinetix experiments (JAX / Flax NNX). Forked from [`real-time-chunking-kinetix`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix). Adds VLASH and A2C2, alongside other modifications. See [`bt-kinetix/README.md`](https://github.com/TheAyos/bt-kinetix#readme) for installation, training, and reproduction. |
| [`bt-libero/`](bt-libero/)             | LIBERO experiments (PyTorch / LeRobot v0.4.2 / SmolVLA). Forked from [`vlash`](https://github.com/mit-han-lab/vlash). Adds IT-RTC, TT-RTC, and A2C2 to the existing VLASH implementation. See [`bt-libero/README.md`](https://github.com/TheAyos/bt-libero#readme). |
| [`presentation/`](presentation/)       | LaTeX source and PDF of the thesis defense slides ([direct PDF link](presentation/Slides%20-%20Understanding%20Asynchronous%20Inference%20Methods%20for%20Vision-Language-Action-Models%20-%20Ayoub%20Agouzoul.pdf)). |
| `report/` *(soon)*              | The full manuscript will be added here.        |

To clone the repository together with both experimental codebases:

```bash
git clone --recurse-submodules https://github.com/TheAyos/async-vla-inference.git
```

If you have already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Acknowledgements

This work would not have been possible without the support and contributions of many people and projects.

- I am sincerely grateful to my supervisor, **Prof. Wenqi Jiang**, for his guidance, patience, and feedback throughout this project.
- I thank the **ETH Systems Group** for hosting me, providing access to the GPU resources used for all experiments, and for the welcoming research environment.
- I thank **École polytechnique** and **ETH Zürich** for the academic and administrative support that enabled this work.

This work stands entirely on the shoulders of remarkable open-source releases. I am deeply grateful to the authors and maintainers of the following projects, which the codebases in this repository are built on:

- [**`real-time-chunking-kinetix`**](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) (Physical Intelligence) — the JAX simulation, base flow-matching policy, and IT-RTC / TT-RTC implementations that `bt-kinetix` extends.
- [**`vlash`**](https://github.com/mit-han-lab/vlash) (MIT HAN Lab) — the VLASH implementation, LIBERO evaluation and SmolVLA training stack that `bt-libero` extends.
- [**`a2c2-libero`**](https://github.com/k1000dai/a2c2-libero) and [**`a2c2-kinetix`**](https://github.com/k1000dai/a2c2-kinetix) — the A2C2 LIBERO and Kinetix reference implementations.
- [**LIBERO**](https://github.com/Lifelong-Robot-Learning/LIBERO) — the manipulation benchmark suite.
- [**Kinetix**](https://github.com/FLAIROx/Kinetix) — the JAX-based 2D physics suite.
- [**LeRobot**](https://github.com/huggingface/lerobot) and [**SmolVLA**](https://huggingface.co/lerobot/smolvla_base) (HuggingFace) — the LIBERO-side training/eval framework and the public VLA backbone used for all LIBERO experiments.

## License

This repository is released under the [MIT License](LICENSE). The submodules retain their respective upstream licenses; please consult them inside each submodule.
