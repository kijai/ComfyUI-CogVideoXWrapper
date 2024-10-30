class PABConfig:
    def __init__(
        self,
        steps: int,
        cross_broadcast: bool = False,
        cross_threshold: list = None,
        cross_range: int = None,
        spatial_broadcast: bool = False,
        spatial_threshold: list = None,
        spatial_range: int = None,
        temporal_broadcast: bool = False,
        temporal_threshold: list = None,
        temporal_range: int = None,
        mlp_broadcast: bool = False,
        mlp_spatial_broadcast_config: dict = None,
        mlp_temporal_broadcast_config: dict = None,
    ):
        self.steps = steps

        self.cross_broadcast = cross_broadcast
        self.cross_threshold = cross_threshold
        self.cross_range = cross_range

        self.spatial_broadcast = spatial_broadcast
        self.spatial_threshold = spatial_threshold
        self.spatial_range = spatial_range

        self.temporal_broadcast = temporal_broadcast
        self.temporal_threshold = temporal_threshold
        self.temporal_range = temporal_range

        self.mlp_broadcast = mlp_broadcast
        self.mlp_spatial_broadcast_config = mlp_spatial_broadcast_config
        self.mlp_temporal_broadcast_config = mlp_temporal_broadcast_config
        self.mlp_temporal_outputs = {}
        self.mlp_spatial_outputs = {}

class CogVideoXPABConfig(PABConfig):
    def __init__(
        self,
        steps: int = 50,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [100, 850],
        spatial_range: int = 2,
        temporal_broadcast: bool = False,
        temporal_threshold: list = [100, 850],
        temporal_range: int = 4,
        cross_broadcast: bool = False,
        cross_threshold: list = [100, 850],
        cross_range: int = 6,
    ):
        super().__init__(
            steps=steps,
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_range=spatial_range,
            temporal_broadcast=temporal_broadcast,
            temporal_threshold=temporal_threshold,
            temporal_range=temporal_range,
            cross_broadcast=cross_broadcast,
            cross_threshold=cross_threshold,
            cross_range=cross_range

        )