
from abc import ABC, abstractmethod
from stok.util.paths import get_reports_dir
from typing import ClassVar
from pathlib import Path

from pydantic import BaseModel

class ExperimentBaseResult(BaseModel):
    ...

class ExperimentBase(ABC):
    """Base class for all experiments"""
    
    NAME: ClassVar[str] = "ExperimentBase"
    VERSION: ClassVar[str] = "v1"
    
    def __init__(self) -> None:
        """Initialize the experiment."""
        pass
    
    @property
    def experiment_name(self) -> str:
        """Get the name of the experiment."""
        return f"{self.NAME}_{self.VERSION}"
    
    @property
    def dump_path(self) -> Path:
        """Get the path to dump the experiment results."""
        return get_reports_dir(self.experiment_name)

    @abstractmethod
    def do(self) -> ExperimentBaseResult:
        """Run the experiment and return results as json to be saved"""
        pass
    
