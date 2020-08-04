#!/bin/bash
blender_path=blender
features_script_path=get_features.py
#feature2_script_path=get_features_2.py
#coverage_script_path=CalculateCoverage3.py
semples_dir=samples/
#scene_1_dir=MoodyBlender
#scene_4_dir=MoodyBlender2
scene_1=Ana
scene_2=Bambo_House
scene_3=BlenderBattery
scene_4=BRZ4
scene_5=MoodyBlender
scene_6=ShipMoscow
servers=ironman
#timese=1
#resolute=20
for sce:qne in $scene_1 $scene_2 $scene_3 $scene_4 $scene_5 $scene_6
#for scene in $scene_1
do for resolut in 20 40 60 80 100
   do for fov in 0 0.105 0.21 0.315 0.42
      do for duplic in 0 1 2 3 4
         do $blender_path --background --python "$features_script_path" "$semples_dir""$scene" "$scene" "$servers" "$resolut" "$fov" "$duplic"
         #$blender_path --background --python "$features_script_path" "$semples_dir""$scene_4_dir" "$scene"_"$times"time "$resolut"
         done
      done
   done
done
