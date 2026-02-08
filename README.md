## 🚀 Deploy on Modal

### 1. Prerequisites
Ensure you have the following **Secrets** configured in your [Modal Dashboard](https://modal.com/secrets):
* `huggingface-secret`
* `groq-secret`

### 2. Quick Start
Run the following commands to clone the `modal_deploy` branch, sync your environment, and launch the app:

```bash
# Clone the repository and enter the directory
git clone -b modal_deploy [https://github.com/YOUR_USERNAME/Gym-Trainer.git](https://github.com/YOUR_USERNAME/Gym-Trainer.git)
cd Gym-Trainer

# Initialize the environment and sync dependencies
uv venv
uv sync
```
#### Start the app in development mode


```bash
modal serve src.app
```
#### OR: Deploy to a permanent URL

```bash
modal deploy src.app
```