
from opencood.tools.atmos_models import LISA

def snow_operation(pcd_path, snow_rate):
    pcd_np = \
            pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='snow')

    atoms_np = atmos_noise.augment(pcd_np, snow_rate)[:, :-1]
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)


def rain_operation(pcd_path, rain_rate):
    pcd_np = \
            pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='rain')

    atoms_np = atmos_noise.augment(pcd_np, rain_rate)[:, :-1]
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)


def fog_operation(pcd_path, visibility):
    pcd_np =  pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='fog')
    atoms_np = atmos_noise.augment(pcd_np, visibility)
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)
    print(f"v = {visibility}")