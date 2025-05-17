
from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar

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
    
    @abstractmethod
    def do(self) -> ExperimentBaseResult:
        """Run the experiment and return results as json to be saved"""
        pass
    
    @property
    def experiment_name(self) -> str:
        """Get the name of the experiment."""
        return f"{self.NAME}_{self.VERSION}"

