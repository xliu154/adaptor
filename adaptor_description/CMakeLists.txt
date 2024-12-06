cmake_minimum_required(VERSION 2.8.3)

project(adaptor_description)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/jacobian_pinv.py
  scripts/vel_control.py
  scripts/jacobian_pinv_arm.py
  scripts/vel_control_arm.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
