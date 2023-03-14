#ENPM-661 project 2, Path planning for point robot using Dijkstra Algorithm


#Importing all the required packages
import cv2
import time
import math
import numpy as np
import heapq


#function to move point robot upwards
def move_up(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    i -= 1
    new_robot_loc = (i, j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)

#function to move point robot downwards
def move_down(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    i += 1
    new_robot_loc = (i, j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)

#function to move point robot right
def move_right(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    j += 1
    new_robot_loc = (i, j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)

#function to move point robot left
def move_left(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    j -= 1
    new_robot_loc = (i, j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)


#function to move robot up and right
def move_upright(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    i -= 1
    j += 1
    new_robot_loc = (i,j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)

#function to move robot up and left
def move_upleft(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    i -= 1
    j -= 1
    new_robot_loc = (i,j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)

#function to move robot down and left
def move_downleft(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    i += 1
    j -= 1
    new_robot_loc = (i, j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)

#function to move robot down and right
def move_downright(robot_loc):
    i = robot_loc[0]
    j = robot_loc[1]
    i += 1
    j += 1
    new_robot_loc = (i, j)
    if i >= 0 and j >= 0:
        return (new_robot_loc, True)
    else:
        return (robot_loc, False)


#function to define the space for the triangle
def in_triangle(x,y):

    x2, y2 = 460, 225
    x3, y3 = 460, 25
    x1, y1 = 510, 125

    area_triangle = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

    area_triangle1 = abs((x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2)
    area_triangle2 = abs((x2 * (y3 - y) + x3 * (y - y2) + x * (y2 - y3)) / 2)
    area_triangle3 = abs((x3 * (y1 - y) + x1 * (y - y3) + x * (y3 - y1)) / 2)

    if area_triangle == area_triangle1 + area_triangle2 + area_triangle3:
        print('tri')
        return True
    else:
        print("The point is outside the triangle.")

#function to define the space of hexagon
def in_hexagon(x, y):

    offset = 0
    p1_x = 300
    p1_y = 200

    p2_x = 365
    p2_y = 163

    p3_x = 365
    p3_y = 88

    p4_x = 300
    p4_y = 50

    p5_x = 235
    p5_y = 88

    p6_x = 235
    p6_y = 162.5

    m1 = (p2_y - p1_y) / (p2_x - p1_x)
    c1 = -m1 * p1_x + p1_y
    t1 = math.atan(m1)
    c1_off =c1 + offset / math.cos(t1)

    m3 = (p4_y - p3_y) / (p4_x - p3_x)
    c3 = -m3 * p3_x + p3_y
    t3 = math.atan(m3)
    c3_off =c3 - offset / math.cos(t3)

    m4 = (p5_y - p4_y) / (p5_x - p4_x)
    c4 = -m4 * p4_x + p4_y
    t4 = math.atan(m4)
    c4_off =c4 - offset / math.cos(t4)

    m6 = (p6_y - p1_y) / (p6_x - p1_x)
    c6 = -m6 * p6_x + p6_y
    t6 = math.atan(m6)
    c6_off =c6 + offset / math.cos(t6)

    InHex = (y >=m3*x +c3_off and y >=m4*x +c4_off and x >= 230 and x <= 370 and y <=m1*x +c1_off and y <=m6*x +c6_off)
    return InHex

#function to define all the possible paths of the robot
def robot_path_graph(init, size_x, size_y):
    x = init[0]
    y = init[1]
    
    if x < size_x and y < size_y:
        path_graph = {}

        if x == size_x - 1 and y == size_y - 1:
            path_graph[(x, y)] = {(x - 1, y), (x - 1, y - 1), (x, y - 1)}

        elif x == size_x - 1 and y == 0:
            path_graph[(x, y)] = {(x - 1, y), (x - 1, y + 1), (x, y + 1)}

        elif y == size_y - 1 and x == 0:
            path_graph[(x, y)] = {(x, y - 1), (x + 1, y - 1), (x + 1, y)}
        
        elif x == 0 and y == 0:
            path_graph[(x, y)] = {(x + 1, y + 1), (x + 1, y), (x, y + 1)}
            
        elif y == 0 and x != 0 and x != size_x - 1:
            path_graph[(x, y)] = {(x - 1, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1)}
        
        elif x == 0 and y != 0 and y != size_y - 1:
            path_graph[(x, y)] = {(x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)}

        elif x == size_x - 1 and y != 0 and y != size_y - 1:
            path_graph[(x, y)] = {(x, y - 1), (x, y + 1), (x - 1, y - 1), (x - 1, y), (x - 1, y + 1)}

        elif y == size_y - 1 and x != 0 and x != size_x - 1:
            path_graph[(x, y)] = {(x - 1, y), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x - 1, y - 1)}

        else:
            path_graph[(x, y)] = {(x - 1, y), (x - 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1),(x, y - 1), (x, y + 1)}

        return (path_graph)
    else:
        pass

#function to calculate the costs of the possible moves
def move_costs(dict, init):
    cost_dict = {}
    for node, value in dict.items():
        cost_dict[node] = {}
        for move in value:

            R = move_right(node)
            L = move_left(node)
            U = move_up(node)
            D = move_down(node)
            UL = move_upleft(node)
            UR = move_upright(node)
            DL = move_downleft(node)
            DR = move_downright(node)

            if (move == R[0]) or (move == L[0]) or (move == U[0]) or (move == D[0]):
                cost_dict[node][move] = 1

            elif (move == UL[0]) or (move == UR[0]) or (move == DL[0]) or (move == DR[0]):
                cost_dict[node][move] = 1.4
    return (cost_dict)


all_paths = {}
backtrack_dict = {}
traversed = []
visit = 0

#the main function to calculate the algorithm
def dijkstra_algorithm(dict, init):

    global visit
    global traversed

    all_paths[init] = 0

    traversed.append(init)

    for pos, val in dict.items():
        all_paths[pos] = math.inf

    open_list = [(0, init)]

    while len(open_list) > 0 and visit == 0:

        next_dist, next_vert = heapq.heappop(open_list)

        if next_dist > all_paths[next_vert]:
            continue
        for move, cost in dict[next_vert].items():

            distance = next_dist + cost

            if distance < all_paths[move]:
                backtrack_dict[move] = {}
                backtrack_dict[move][distance] = next_vert
                all_paths[move] = distance
                heapq.heappush(open_list, (distance, move))

                if move not in traversed:
                    traversed.append(move)

                    if move == final:
                        print('GOAL REACHED')
                        visit = 1
                        break

    return (all_paths, traversed, backtrack_dict)


#function to unpack the shortest path
def back_track_path(backtrack_dict, final, init):
    backtrack_list = []
    backtrack_list.append(init)
    while final != []:
        for key, value in backtrack_dict.items():
            for key2, val2 in value.items():
                if key == init:
                    if val2 not in backtrack_list:
                        backtrack_list.append(init)
                    init = val2
                    if val2 == final:
                        final = []
                        break
    return (backtrack_list)


#function to create the configuration space for the robot
def robot_space(max_x, max_y, init, final):

    max_x += 1
    max_y += 1

    config_space = []
    for i in range(0, 601):
        for j in range(0,251):
            config_space.append((i, j))
    print('Length of config_space')
    print(len(config_space))

    obstacle_space = []
    for c in config_space:
        x = c[0]
        y = c[1]

        if (x >= 100 and x <= 150 and y >= 0 and y <= 100):
            obstacle_space.append((x, y))

        if (x >= 100 and x <= 150 and y >= 150 and y <= 250):
            obstacle_space.append((x, y))

        if is_inside_hexagon(x, y):
            obstacle_space.append((x, y))
            print('hex found')

        if in_triangle(x, y):
            obstacle_space.append((x, y))


    if final in obstacle_space:
        print('The final is Invalid, please exit and try again')

    original_graph = {}
    for i in range(max_x - 1, -1, -1):
        for j in range(max_y - 1, -1, -1):
            dict = robot_path_graph((i, j), max_x, max_y)
            original_graph[(i, j)] = dict[(i, j)]

    print('Length of original_space')
    print(len(original_graph))

    for key, value in original_graph.items():
        value_copy = value.copy()
        for coordinates in value_copy:
            if coordinates in obstacle_space:
                value.remove(coordinates)
    print('detecting obstacles')
    original_graph_copy = original_graph.copy()
    for key, value in original_graph_copy.items():
        if key in obstacle_space:
            del original_graph[key]
    print('deleted obstacle spaces')
    print('Length of required_space')
    print(len(original_graph))

#calculating all the costs of the possible moves of the robot
    costs_calculated = move_costs(original_graph, init)
    req_graph = costs_calculated

#implementing the dijsktra algorith with the robot space and the obstacles
    all_paths, traversed, backtrack_dict = dijkstra_algorithm(req_graph, init)
    all_distance_copy = all_paths.copy()
    for key, value in all_distance_copy.items():
        if all_distance_copy[k] == math.inf:
            del all_paths[k]
    return (all_paths, traversed, backtrack_dict, obstacle_space)


#take the input form the user
init = tuple(map(int, input("Enter the X and Y initial coordinates(x,y): ").split(',')))
final = tuple(map(int, input("Enter the X and Y final coordinates (x,y): ").split(',')))

#defining the maximum frame sizes
max_x = 600
max_y = 250

#initializing time to track the processing time
init_time = time.time()
#calling the function to setup the robot configuration space
all_paths, traversed, backtrack, total_points = robot_space(max_x, max_y, init, final)

#calling the backtrack function
rob_backtrack = back_track_path(backtrack, init, final)
print("Shortest path", rob_backtrack)
print("Time taken for shortest path : ", time.time() - init_time, "seconds")

#creating a blank canvas
display_canvas = np.zeros((251, 601, 3), np.uint8)

#defining a color to all the non-obstacle spaces
for c in total_points:
    i = c[1]
    j = c[0]
    display_canvas[(i, j)] = [255, 0, 255]

#flipping the graph with respect to y-axis
display_canvas = np.flipud(display_canvas)

#copying the canvas to display all paths and shortest path
display_canvas_copy_backtrack = display_canvas.copy()
display_canvas_copy_visited = display_canvas.copy()

cv2.imshow('display_canvas', display_canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

#display the traverse path and write the video
out = cv2.VideoWriter('dijkstra.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (600, 250))
for pos in traversed:
    i = pos[0]
    j = pos[1]
    j = int(250 - j)
    cv2.rectangle(display_canvas_copy_visited, (i, j), (i + 1, j - 1), (0, 0, 255), -1)
    out.write(display_canvas_copy_visited)
    cv2.imshow('Robot', display_canvas_copy_visited)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

out.release()
cv2.destroyAllWindows()

#Redefining the coordinates to display the traversed path.
for pos in traversed:
    i = pos[0]
    j = pos[1]
    display_canvas_copy_backtrack[(250 - j, i)] = [255, 0, 0]

#Resizing and displaying the image
res_rob_bactrack = cv2.resize(display_canvas_copy_backtrack, (1200, 500))
cv2.imshow('traversed', res_rob_bactrack)
cv2.waitKey(0)
cv2.imwrite('traversed.jpg', res_rob_bactrack)
cv2.destroyAllWindows()

#Redifing the coordinates to display the shortest path
for pos in rob_backtrack:
    i = pos[0]
    j = pos[1]
    display_canvas_copy_backtrack[(250 - j, i)] = [0, 255, 0]

#Resizing and displaying the shortest path
res_rob_bactrack = cv2.resize(display_canvas_copy_backtrack, (1200, 500))
cv2.imshow('res_rob_bactrack', res_rob_bactrack)
cv2.waitKey(0)
cv2.imwrite('backtacked.jpg', res_rob_bactrack)
cv2.destroyAllWindows()




