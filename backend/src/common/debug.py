import os
import logging
import sys

logger = logging.getLogger(__name__)

def enable_remote_debugging(port: int):
    """
    Enables remote debugging with debugpy if the ENABLE_DEBUGGING environment variable is set.
    """
    if os.environ.get("ENABLE_DEBUGGING", "").lower() in ("1", "true", "yes", "on"):
        try:
            import debugpy
            # 5678 is the default port for debugpy, but we allow overriding it
            # to support multiple services on the same host.
            debugpy.listen(("0.0.0.0", port))
            logger.info(f"Remote debugging enabled on port {port}")
            # Optional: wait for debugger to attach
            # debugpy.wait_for_client() 
        except ImportError:
            logger.error("debugpy not installed, cannot enable remote debugging")
        except Exception as e:
            logger.error(f"Failed to enable remote debugging: {e}")
