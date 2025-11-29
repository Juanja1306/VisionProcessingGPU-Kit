from abc import ABC, abstractmethod
from .types.index import RGBImage

class AplicationFilter(ABC):
    
    @abstractmethod
    async def aplyFilter(self) -> RGBImage:
        pass