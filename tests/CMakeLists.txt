add_executable(t_raycasting test_raycasting.cpp)
target_link_libraries(t_raycasting RCasting Eigen3::Eigen)

add_executable(t_intersection test_intersection.cpp)
target_link_libraries(t_intersection RCasting Eigen3::Eigen)