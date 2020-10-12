weight = [0.33153257, 0.33853987, 0.3299275]

alpha_lin = [23.819906, 36.027424]
beta_lin = [25.32551, 4.589774]
alpha_ang = [25.2909, 53.92586]
beta_ang = [25.234571, 51.74721]

a_lin_MCP = (weight[0]*alpha_lin[0] + weight[1]*alpha_lin[1]) / (weight[0] + weight[1])
b_lin_MCP = (weight[0]*beta_lin[0] + weight[1]*beta_lin[1]) / (weight[0] + weight[1])
a_ang_MCP = (weight[0]*alpha_ang[0] + weight[2]*alpha_ang[1]) / (weight[0] + weight[2])
b_ang_MCP = (weight[0]*beta_ang[0] + weight[2]*beta_ang[1]) / (weight[0] + weight[2])

mode_lin = (a_lin_MCP - 1) / (a_lin_MCP + b_lin_MCP - 2)
mode_ang = (a_ang_MCP - 1) / (a_ang_MCP + b_ang_MCP - 2)
print(mode_lin, mode_ang)