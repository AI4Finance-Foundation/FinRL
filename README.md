# finrl-library

FinRL library, a DRL library designed specifically for automated stock trading with an effort to close sim-real gap.

**Table of contents:**

- [Status](#status)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Usage](#usage)

## Status: Release
<details><summary><b>Current status</b> <i>[click to expand]</i></summary>
<div>
We are currently open to any suggestions or pull requests from the community to make RLzoo a better repository. Given the scope of this project, we expect there could be some issues over
the coming months after initial release. We will keep improving the potential problems and commit when significant changes are made in the future. Current default hyperparameters for each algorithm and each environment may not be optimal, so you can play around with those hyperparameters to achieve best performances. We will release a version with optimal hyperparameters and benchmark results for all algorithms in the future.
</div>
</details>

<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 1.0.3 (Current version)

  Changes:

  * Fix bugs in SAC algorithm

* 1.0.1

	Changes:
	* Better support RLBench environment, with multi-head network architectures to support dictionary as observation type;
	* Make the code cleaner.
* 0.0.1
</div>
</details>

## Installation
Ensure that you have **Python >=3.6**

Direct installation:
```
pip3 install finrl --upgrade
```
Install finrl-library from Git:
```
git clone https://github.com/finrl/finrl-library.git
cd finrl-library
pip3 install .
```

## Prerequisites
```
pip3 install -r requirements.txt
```
<details><summary><b>List of prerequisites.</b> <i>[click to expand]</i></summary>
<div>

* tensorflow >= 1.14

</div>
</details>



---

## Contributors

- Hongyang Yang <hy2500@columbia.edu>

---

## License & copyright

© Hongyang Yang

Licensed under the [MIT License](LICENSE).
