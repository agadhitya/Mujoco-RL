# YCB Object Spawner
Spawns objects from the YCB Dataset located within the 'desc' folder

## Simple Spawning
If any object needs to be specifically spawned, you can use the file located in desc/*.urdf:

'rosrun gazebo_ros spawn_model ...'

Do note that initial poses can also be specified within gazebo

## Using object_scene_spawner
If a more powerful interface is needed, there is an automatic object spawner node that is available.  This node relies on a set of parameters to be supplied for each object to be spawned.  The parameter file is organized as such:
```
ycb_models:
  names:
    - name 1
    - name 2
  poses:
    name 1:
      position: [REQUIRED]
        x: X
        y: Y
        z: Z
      orientation: [OPTIONAL]
        x: X
        y: Y
        z: Z
        w: W
    name 2:
      position: [REQUIRED]
        x: X
        y: Y
        z: Z
```
Note that the orientation is optional.  If none is given, it defaults to {0, 0, 0, 1} (x, y, z, w)

For every object {name}, there MUST be at LEAST a corresponding position under poses/{name}.  If that does not exist then the YCBSpawnerNode will not spawn that object

In addition, once executed and the relevant objects are spawned, the ycb object urdfs will be placed on the parameter server under 'ycb_models/urdfs/{NAME}'

To actually run the spawner, simply run it with no arguments supplied: 'rosrun ycb_models object_scene_spawner'
## Description Uploading
If an application requires the urdfs to be on the parameter server, then running the object_description_uploader node can accomplish that

Same as the method shown above, the parameters it requires are:
```
ycb_models:
  names:
    - name 1
    - name 2
```
Once it reads those, it will look for the URDFS and upload them to the ycb_models/urdfs/{NAME}

To actually run it, simply run it with no arguments supplied: 'rosrun ycb_models object_description_uploader'