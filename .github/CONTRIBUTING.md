# Contributing to LLaMA Factory

Everyone is welcome to contribute, and we value everybody's contribution. Code contributions are not the only way to help the community. Answering questions, helping others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts about the awesome projects it made possible, shout out on Twitter every time it has helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our [code of conduct](CODE_OF_CONDUCT.md).

**This guide was heavily inspired by [transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to LLaMA Factory:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Contribute to the examples or to the documentation.

### Style guide

LLaMA Factory follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), check it for details.

### Create a Pull Request

1. Fork the [repository](https://github.com/hiyouga/LLaMA-Factory) by clicking on the [Fork](https://github.com/hiyouga/LLaMA-Factory/fork) button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

```bash
git clone git@github.com:[username]/LLaMA-Factory.git
cd LLaMA-Factory
git remote add upstream https://github.com/hiyouga/LLaMA-Factory.git
```

3. Create a new branch to hold your development changes:

```bash
git checkout -b dev_your_branch
```

4. Set up a development environment by running the following command in a virtual environment:

```bash
pip install -e . -r requirements/dev.txt
```

If LLaMA Factory was already installed in the virtual environment, remove it with `pip uninstall llamafactory` before reinstalling it in editable mode with the -e flag.

5. Check code before commit:

```bash
make commit
make style && make quality
make test
```

6. Submit changes:

```bash
git add .
git commit -m "commit message"
git fetch upstream
git rebase upstream/main
git push -u origin dev_your_branch
```

7. Create a merge request from your branch `dev_your_branch` at [origin repo](https://github.com/hiyouga/LLaMA-Factory).
