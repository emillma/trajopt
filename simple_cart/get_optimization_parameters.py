from equations import second_order_system


system = second_order_system()
system.set_static_values([9.81, 1])

N = 3
end_time = 3
system.set_grid_size(N, end_time)
system.set_boundary_constraints([0, 0, 0, 1], [1, 0, end_time, 1])
system.set_path_constraints([None, None, None, None],
                            [None, None, None, None],

                            [None, 0],
                            [None, 0])
system.set_collocation_constraints()
system.set_cost_function()

(cost_lamda, initial_args,
 cost_jacobian_lambda, constraints) = system.get_optimization_parameters()
