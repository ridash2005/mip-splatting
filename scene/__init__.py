#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](args.source_path, args.white_background, args.eval, args.load_allres)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # The stress suite (§6.4) degrades the CAPTURE, not the scene: training
        # cameras are restricted to a protocol's index list while the test set is
        # left whole, so the only thing that varies between protocols is the
        # geometry the reconstruction had available.
        subset = getattr(args, "camera_subset", "")
        if subset:
            import json as _json
            from PIL import Image
            from utils.graphics_utils import fov2focal
            spec = _json.loads(subset) if subset.strip().startswith(("[", "{")) \
                else _json.load(open(subset))
            focal = None
            if isinstance(spec, dict):
                idx, focal = spec["idx"], spec.get("focal")
            else:
                idx = spec
            before = len(scene_info.train_cameras)
            picked = [scene_info.train_cameras[i] for i in idx]

            # `mixed_focal` is the one protocol that changes INTRINSICS rather
            # than pose: every camera is kept and a seeded random half is
            # downsampled. Restricting the index list alone leaves it identical
            # to the full orbit -- the protocol would then be recorded under a
            # name for a capture it is not, which is precisely the defect §7.2
            # exists to warn about.
            #
            # Downsampling by k scales the focal length AND the image dimensions
            # by 1/k, so the field of view is UNCHANGED and what changes is the
            # pixel footprint -- which is the whole point, since f_k takes
            # d_min and f_max from different cameras. Overriding the FoV instead
            # would point the camera somewhere else and render a different scene.
            if focal:
                assert len(focal) == len(idx), (
                    f"camera_subset: {len(focal)} focal lengths for "
                    f"{len(idx)} cameras")
                changed = 0
                for j, cam in enumerate(picked):
                    k = fov2focal(cam.FovX, cam.width) / float(focal[j])
                    if abs(k - 1.0) < 1e-6:
                        continue
                    w, h = max(1, round(cam.width / k)), max(1, round(cam.height / k))
                    picked[j] = cam._replace(
                        image=cam.image.resize((w, h), Image.LANCZOS),
                        width=w, height=h)
                    changed += 1
                print(f"camera_subset: downsampled {changed} of {len(picked)} "
                      f"cameras; field of view unchanged, pixel footprint is not")

            scene_info = scene_info._replace(train_cameras=picked)
            print(f"camera_subset: training on {len(idx)} of {before} cameras")

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]