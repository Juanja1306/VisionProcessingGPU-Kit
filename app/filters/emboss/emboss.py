from dataclasses import dataclass

from .abstractEmboss import AplicationFilter


@dataclass(frozen=True)
class EmbossProps:
    serviceEmboss: AplicationFilter

class Emboss:
    def __init__(self, options: EmbossProps):
        self.__serviceEmboss = options.serviceEmboss
    
    async def aplyFilter(self):
        return await self.__serviceEmboss.aplyFilter()