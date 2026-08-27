from setuptools import find_packages, setup

package_name = 'suas_auto'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='youssef',
    maintainer_email='youssefgamer037@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'takeoff_sitl = suas_auto.takeoff_sitl:main',
            'waypoint_sitl = suas_auto.waypoint_sitl:main',
            'takeoff_land = suas_auto.takeoff_land_sitl:main',
            'payload_sitl = suas_auto.payload_sitl:main',
            'lawnmower = suas_auto.lawnmower_sitl:main',
            'lawnmower_quad = suas_auto.lawnmower_quad:main',
            'waypoint_lawnmower = suas_auto.waypoint_lawnmower_sitl:main',
            'vision_detect = suas_auto.vision_detect_node:main',
        ],
    },
)
