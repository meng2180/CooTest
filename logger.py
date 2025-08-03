import logging
import os.path


class CLogger:
    # filename = os.path.join(config.common_config.project_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,

        format='%(asctime)s - %(levelname)s: %(message)s',

    )

    logging.getLogger('Console')

    @staticmethod
    def set_level(level):
        logging.basicConfig(level=level,
                            format='%(asctime)s - %(levelname)s: %(message)s',

                            )

    @staticmethod
    def info(message: str):
        logging.info("\033[0;32m" + message + "\033[0m")

    @staticmethod
    def warning(message: str):
        logging.warning("\033[0;33m" + message + "\033[0m")

    @staticmethod
    def error(message: str):
        logging.error("\033[0;31m" + "-" * 120 + '\n| ' + message + "\033[0m" + "\n" + "└" + "-" * 150)

    @staticmethod
    def debug(message: str):
        logging.debug("\033[0;37m" + message + "\033[0m")
