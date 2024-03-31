import heapq
from itertools import combinations, product

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def move_joint_state(locs, dir):

    new_locs = []
    for i in range(len(locs)):
        new_locs.append((locs[i][0] + dir[i][0], locs[i][1] + dir[i][1]))
    return new_locs

def generate_motions_recursive(num_agents,cur_agent):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    joint_state_motions = list(product(directions, repeat = num_agents))
    #joint_state_motions = list(combinations_with_replacement(directions, num_agents))
    return joint_state_motions


def is_valid_motion(old_loc, new_loc):
    ##############################
    # Task 1.3/1.4: Check if a move from old_loc to new_loc is valid
    # Check if two agents are in the same location (vertex collision)
    #
    comb_vert = combinations(new_loc, 2)
    for i in list(comb_vert):
        if i[0] == i[1]:
            return False
    # Check edge collision
    #
    edges = list(zip(old_loc, new_loc))
    comb_edge = combinations(edges, 2)
    for i in list(comb_edge):
        if i[0][0] == i[1][1] and i[0][1] == i[1][0]:
            return False
    return True

def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent, goal_loc):
    ##############################
    # Task 1.2/1.3/1.4: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    constraint_table = {}
    inf_constraints = {}
    goal_constraint = None
    
    def add_to_constraint_table(timestep, constraint_loc, constraint_type):
        constraint = {'loc': constraint_loc, 'type': constraint_type}

        if timestep in constraint_table:
            constraint_table[timestep].append(constraint)
        else:
            constraint_table[timestep] = []
            constraint_table[timestep].append(constraint)

    def add_to_inf_constraints(timestep, constraint_loc, constraint_type):
        inf_constraints[constraint_loc] = {'timestep': timestep, 'type': constraint_type}

    for constraint in constraints:
        if constraint['agent'] == agent:
            timestep = constraint['timestep']
            constraint_loc = constraint['loc']
            if 'type' in constraint:
                constraint_type = constraint['type']
            else:  # For backwards compatibility
                num_locs = len(constraint['loc'])
                constraint_type = 'vertex' if num_locs == 1 else 'edge' if num_locs == 2 else None

            add_to_constraint_table(timestep, constraint_loc, constraint_type)

            if constraint_type == 'inf':
                add_to_inf_constraints(timestep, constraint_loc, constraint_type)

            if constraint_type == 'vertex' and constraint_loc[0] == goal_loc:
                goal_constraint = timestep if goal_constraint is None or goal_constraint < timestep else goal_constraint

    return constraint_table, inf_constraints, goal_constraint


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3/1.4: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    if next_time in constraint_table:
        constraints = constraint_table[next_time]  # list of constraints on the agent
                                                   # at the this timestep
        for constraint in constraints:
            if constraint['type'] == 'vertex' and next_loc == constraint['loc'][0]:
                # vertex constraint violated
                return True
            elif constraint['type'] == 'edge' and \
                    curr_loc == constraint['loc'][0] and next_loc == constraint['loc'][1]:
                # edge constraint violated
                return True

    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True

def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True

"""def not_all_in_obstacles(map, locs):
    for loc in locs:
        if not map[loc[0]][loc[1]]:
            return False
    return True"""

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.2/1.3/1.4: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.
    def is_goal(node):
        if node['loc'] == goal_loc and (goal_constraint is None or node['timestep'] > goal_constraint):
            return True
        
        return False
    
    def exceeds_time_limit(node):
        map_spots = 0
        for row in my_map:
            map_spots += row.count(False)

        timesteps_max = map_spots #+ len(constraints)
        if node['timestep'] > timesteps_max:
            return True   
    
    constraint_table, inf_constraints, goal_constraint = build_constraint_table(constraints, agent, goal_loc)
    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep': 0}#edit
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 2.2: Adjust the goal test condition to handle goal constraints
        if is_goal(curr):
            return get_path(curr)
        
        if exceeds_time_limit(curr):
            print('------TIME LIMIT EXCEEDED------')
            return None
        
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if not in_map(my_map, child_loc) or my_map[child_loc[0]][child_loc[1]]:
                continue
            
            if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue

            if child_loc in inf_constraints:
                if curr['timestep'] + 1 >= inf_constraints[child_loc]['timestep']:
                    continue
            
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep': curr['timestep'] + 1}
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions


def joint_state_a_star(my_map, starts, goals, h_values, num_agents):
    """ my_map      - binary obstacle map
        start_loc   - start positions
        goal_loc    - goal positions
        num_agent   - total number of agents in fleet
    """

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = 0
     ##############################
    # Task 1.1: Iterate through starts and use list of h_values to calculate total h_value for root node - COMPLETED
    #
    # 
    for i in range(num_agents):
        start_loc = starts[i]
        h_value += h_values[i][start_loc]
    
    root = {'loc': starts, 'g_val': 0, 'h_val': h_value, 'parent': None }
    push_node(open_list, root)
    closed_list[tuple(root['loc'])] = root

     ##############################
    # Task 1.1:  Generate set of all possible motions in joint state space - COMPLETED
    #
    # 
    directions = generate_motions_recursive(num_agents,0)
    while len(open_list) > 0:
        curr = pop_node(open_list)

        if curr['loc'] == goals:
            return get_path(curr)

        for dir in directions:
            ##############################
            # Task 1.1:  Update position of each agent - COMPLETED
            #
            # 
            child_loc = move_joint_state(curr['loc'], dir)
            if not all_in_map(my_map, child_loc):
                continue
             ##############################
            # Task 1.1:  Check if any agent is in an obstacle - COMPLETED
            #
            #valid_move = not_all_in_obstacles(my_map, child_loc)
            # 
            valid_move = True
            for loc in child_loc:
                if my_map[loc[0]][loc[1]]:
                    valid_move = False

            if not valid_move:
                continue

             ##############################
            # Task 1.1:   check for collisions - COMPLETED
            #
            # 
            if not is_valid_motion(curr['loc'],child_loc):
                continue
            
             ##############################
            # Task 1.1:  Calculate heuristic value - COMPLETED
            #
            # 
            h_value = 0
            for i in range(num_agents):
                loc = child_loc[i]
                h_value += h_values[i][loc]

            # Create child node
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + num_agents,
                    'h_val': h_value,
                    'parent': curr}
            if tuple(child['loc']) in closed_list:
                existing_node = closed_list[tuple(child['loc'])]
                if compare_nodes(child, existing_node):
                    closed_list[tuple(child['loc'])] = child
                    push_node(open_list, child)
            else:
                closed_list[tuple(child['loc'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions