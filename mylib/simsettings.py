import habitat_sim

# Function to create the simulator configuration
def make_cfg(settings):
    """
    Description:
        Creates and returns a Habitat simulator configuration using settings 
        provided in a dictionary. This includes simulator, sensor, and agent setup.
    
    Arguments:
        settings (dict): A dictionary containing the following keys:
            - "dataset" (str): Path to the scene dataset config file.
            - "scene" (str): Scene ID or path to the scene file.
            - "enable_physics" (bool): Whether to enable physics simulation.
            - "height" (int): Sensor image height in pixels.
            - "width" (int): Sensor image width in pixels.
            - "sensor_height" (float): Height of the sensor from the agent base.
            - "color_sensor" (bool): Whether to include a color sensor.
            - "depth_sensor" (bool): Whether to include a depth sensor.
    
    Returns:
        habitat_sim.Configuration: The fully constructed simulator configuration.
    """

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_dataset_config_file = settings["dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {"sensor_type": habitat_sim.SensorType.COLOR, "resolution": [settings["height"], settings["width"]], "position": [0.0, settings["sensor_height"], 0.0],},
        "depth_sensor": {"sensor_type": habitat_sim.SensorType.DEPTH, "resolution": [settings["height"], settings["width"]], "position": [0.0, settings["sensor_height"], 0.0],},
    }

    sensor_specs = []
    for sensor_uuid, params in sensors.items():
        if settings[sensor_uuid]:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = sensor_uuid
            spec.sensor_type = params["sensor_type"]
            spec.resolution = params["resolution"]
            spec.position = params["position"]
            sensor_specs.append(spec)

    # NOTE: Not sure we will really need some of this
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])