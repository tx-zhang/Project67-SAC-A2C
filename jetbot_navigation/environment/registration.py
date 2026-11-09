"""Explicit Gym registration for the Isaac environment.

Registration is a function rather than an import side effect so the lightweight
NumPy package can still be imported on machines without Isaac Sim.
"""

TASK_ID = "Isaac-Jetbot-Navigation-Direct-v0"


def register_isaac_env(backend="lab3"):
    """Register once after AppLauncher has initialized Isaac Sim."""
    import gymnasium as gym

    modules = {
        "lab3": "jetbot_navigation.environment.isaac_env",
        "lab211": "jetbot_navigation.environment.isaac_env_211",
    }
    if backend not in modules:
        raise ValueError(f"unknown backend {backend!r}; choose lab3 or lab211")
    task_id = TASK_ID if backend == "lab3" else "Isaac-Jetbot-Navigation-Lab211-Direct-v0"
    module = modules[backend]
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point=f"{module}:JetbotNavigationIsaacEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{module}:JetbotNavigationIsaacEnvCfg"
            },
        )
    return task_id


__all__ = ["TASK_ID", "register_isaac_env"]
