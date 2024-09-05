# Development Environment Setup

Every contributor to this repository should develop in a fork.

```bash
cd ai-edge-quantizer
python -m venv --prompt ai-edge-quantizer venv
source venv/bin/activate

pip install -r dev-requirements.txt
```

## Running Tests

```bash
python -m unittest discover --pattern *_test.py
```

## Code Formatting

You can format your changes with our preconfigured formatting script.

```bash
cd ai-edge-quantizer
bash ./format.sh
```

## Build PyPi Package at Local

```bash
pip install wheel
python setup.py bdist_wheel
```

It will build a PyPi package `ai_edge_quantizer-0.0.1-py3-none-any.whl` unde
the `dist` folder, which you could install at local using command:

```bash
pip install dist/ai_edge_quantizer-0.0.1-py3-none-any.whl
```


# Contributor License Agreement

- Contributions to this project must be accompanied by a [Contributor License
  Agreement](https://cla.developers.google.com/about) (CLA).

- Visit <https://cla.developers.google.com/> to see your current agreements or
  to sign a new one.

# Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

# Code Contribution Guidelines

We recommend that contributors read these tips from the Google Testing Blog:

- [Code Health: Providing Context with Commit Messages and Bug Reports](https://testing.googleblog.com/2017/09/code-health-providing-context-with.html)
- [Code Health: Understanding Code In Review](https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html)
- [Code Health: Too Many Comments on Your Code Reviews?](https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html)
- [Code Health: To Comment or Not to Comment?](https://testing.googleblog.com/2017/07/code-health-to-comment-or-not-to-comment.html)

