from dataclasses import dataclass
from typing import Optional

@dataclass
class Driver:
    """Represents a delivery driver."""
    id: str
    lat: float
    lon: float
    status: str = "IDLE"
    
    # Store the OSM node id nearest to the driver for routing
    node_id: Optional[int] = None

@dataclass
class Package:
    """Represents a package to be delivered."""
    id: str
    lat: float
    lon: float
    status: str = "PENDING"
    
    # Store the OSM node id nearest to the package for routing
    node_id: Optional[int] = None
