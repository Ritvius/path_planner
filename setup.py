from setuptools import find_packages, setup

package_name = 'path_planner'

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
    maintainer='milton',
    maintainer_email='milton.ritzing@gmail.com',
    description='Path planning summer project',
    license='Feel free to share with friends!',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "path_planner_node = path_planner.path_planner_node:main"
        ],
    },
)