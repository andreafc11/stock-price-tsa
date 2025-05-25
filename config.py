# config.py - Configuration management system
import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    window_size: int = 30
    volatility_window: int = 30
    rsi_period: int = 14
    bollinger_std: float = 2.0
    trend_method: str = 'linear'  # 'linear', 'polynomial'
    outlier_method: str = 'iqr'   # 'iqr', 'zscore'
    outlier_threshold: float = 1.5

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figure_size: tuple[int, int] = (16, 10)
    dpi: int = 300
    style: str = 'seaborn-v0_8-darkgrid'
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        'price': '#2E86AB',
        'ma': '#A23B72',
        'ema': '#F18F01',
        'volatility': '#F18F01',
        'trend': '#C73E1D',
        'bollinger': 'red',
        'rsi': '#A23B72',
        'volume': '#2E86AB',
        'outliers': 'red'
    })
    include_volume: bool = True
    include_bollinger: bool = False
    include_rsi: bool = False
    save_format: str = 'png'  # 'png', 'pdf', 'svg'
    
@dataclass
class OutputConfig:
    """Configuration for output settings."""
    output_dir: str = 'output'
    save_plots: bool = False
    save_reports: bool = False
    report_format: str = 'json'  # 'json', 'csv', 'html'
    filename_template: str = '{symbol}_{analysis_type}'
    create_subdirs: bool = True

@dataclass
class AppConfig:
    """Main application configuration."""
    analysis: AnalysisConfig
    visualization: VisualizationConfig
    output: OutputConfig
    logging_level: str = 'INFO'
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create configuration from dictionary."""
        return cls(
            analysis=AnalysisConfig(**config_dict.get('analysis', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            logging_level=config_dict.get('logging_level', 'INFO'),
            verbose=config_dict.get('verbose', False)
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        """Load configuration from file (JSON or YAML)."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration with fallback to default config file locations.
    
    Args:
        config_path: Explicit path to config file
        
    Returns:
        AppConfig: Loaded or default configuration
    """
    # Default config file locations (in order of preference)
    default_locations = [
        'stock_analysis_config.yaml',
        'stock_analysis_config.json',
        os.path.expanduser('~/.stock_analysis_config.yaml'),
        os.path.expanduser('~/.stock_analysis_config.json'),
    ]
    
    if config_path:
        return AppConfig.from_file(config_path)
    
    # Try default locations
    for location in default_locations:
        if os.path.exists(location):
            return AppConfig.from_file(location)
    
    # Return default configuration
    return AppConfig(
        analysis=AnalysisConfig(),
        visualization=VisualizationConfig(),
        output=OutputConfig()
    )

def create_default_config(output_path: str = 'stock_analysis_config.yaml') -> None:
    """Create a default configuration file."""
    default_config = AppConfig(
        analysis=AnalysisConfig(),
        visualization=VisualizationConfig(),
        output=OutputConfig()
    )
    default_config.save(output_path)
    print(f"Default configuration created: {output_path}")