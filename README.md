# Vision-Based Tactile Sensor

## 🔧 Setup

### Option 1: Simple Setup (using `venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Using [`uv`](https://github.com/astral-sh/uv)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync dependencies (this reads `pyproject.toml` and creates a virtual environment):

```bash
uv sync
```

---

## 📈 External Dependency: Natural Helmholtz–Hodge Decomposition

This project uses the [NaturalHHD](https://github.com/bhatiaharsh/naturalHHD) package.

### 🛠️ Manual Installation (Required)

1. Clone the repository under the `external/` directory:

   ```bash
   git clone https://github.com/bhatiaharsh/naturalHHD.git external/naturalHHD
   ```

2. Navigate to the Python implementation:

   ```bash
   cd external/naturalHHD/pynhhd-v1.1
   ```

3. **Fix deprecated code**:
   Open `pynhhd/poisson.py` and replace all instances of `np.float` with `float` to avoid deprecation warnings in NumPy.

4. Install the module:

   Using `pip`:

   ```bash
   pip install -e .
   ```

   Or using `uv`:

   ```bash
   uv pip install -e .
   ```

5. Verify installation:

   ```bash
   pip list | grep pynhhd
   ```

---

## ▶️ Running the Main Script

To run the main application:

```bash
uv run python3 main.py
```

This script reads `data/test_video.mp4`, displays the output in real time, and saves multiple processed video outputs.

### 🔄 TODO

* Replace the static video input with a live stream for real-time tactile sensing.