from functions import * 

# plot edge cells
grid1 = make_grid(max_size=10)
grid_plot([grid1], ['Edge Cells'], 'Edge_Cells.png')

# plot step perimeters
grid2a = make_grid([3,3])
grid2b = make_grid([0,3])
grid2c = make_grid([0,0])
grid_plot([grid2a, grid2b, grid2c], ['Inside', 'Edge', 'Corner'], 
          'Step_Perimeter.png')

# plot view perimeters
grid3a = make_grid([4,4], 0, 10, 'view')
grid3b = make_grid([4,4], 1, 10, 'view')
grid3c = make_grid([4,4], 2, 10, 'view')
grid3d = make_grid([4,4], 3, 10, 'view')
grid_plot([grid3a, grid3b, grid3c, grid3d], 
          ['View Size = 0', 'View Size = 1', 'View Size = 2', 'View Size = 3'], 
          'View_Perimeter.png')

