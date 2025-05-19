import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from fedops.client.manager import FLClientManager


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("📋 FedOps Client Manager Initialized")
    logger.info("Using config:\n" + OmegaConf.to_yaml(cfg))

    client_manager = FLClientManager(cfg)

    client_manager.register_client()

    if client_manager.health_check():
        logger.info("✅ Server is reachable. Starting training...")
        client_manager.start_training()
    else:
        logger.error("❌ Failed to reach server. Exiting.")


if __name__ == "__main__":
    main()
