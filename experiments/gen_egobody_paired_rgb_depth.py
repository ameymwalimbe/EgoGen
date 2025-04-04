# what is needed for a unified script
# what are the things we want to control and vary

#   different scenes, number of data points per scene, same model 

# need to be able to control the number of data points per scene, but otherwise this should work
# also should be adapted to work with multiple scenes

import gen_egobody_depth
import gen_egobody_rgb
import subprocess

if __name__ == '__main__':
    scene_name = "seminar_d78"
    # scene_name = "cab_e"
    # while True:
    #     ret = subprocess.call(['python', 'crowd_ppo/main_egobody_eval.py', '--resume-path=data/checkpoint_best.pth', '--watch', '--scene-name=%s' % scene_name])
    #     if ret == 0:
    #         break
    # gen_egobody_depth.genDepth(100, scene_name)
    gen_egobody_rgb.genRGB(100, scene_name)