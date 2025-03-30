from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    n = data.get('n')
    grid = data.get('grid')  # grid is a 2D list with values: 'start', 'end', 'obstacle', or 'empty'
    gamma = 0.9

    # Set up rewards: normal cell = -1, end = +10, obstacle = -6.
    rewards = [[-1 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'end':
                rewards[i][j] = 10
            elif grid[i][j] == 'obstacle':
                rewards[i][j] = -6

    # Locate the start and end positions.
    start = None
    end = None
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'start':
                start = (i, j)
            if grid[i][j] == 'end':
                end = (i, j)
    if start is None or end is None:
        return jsonify({'error': 'Start or end not set.'}), 400

    # Helper: determine allowed directions for cell (i, j)
    def get_allowed_directions(i, j):
        allowed = ['↑', '↓', '←', '→']
        if i == 0:
            allowed.remove('↑')
        if i == n - 1:
            allowed.remove('↓')
        if j == 0:
            allowed.remove('←')
        if j == n - 1:
            allowed.remove('→')
        # Avoid pointing directly toward the start if adjacent.
        if abs(i - start[0]) + abs(j - start[1]) == 1:
            dx = start[0] - i
            dy = start[1] - j
            if dx == 1 and '↓' in allowed: allowed.remove('↓')
            elif dx == -1 and '↑' in allowed: allowed.remove('↑')
            if dy == 1 and '→' in allowed: allowed.remove('→')
            elif dy == -1 and '←' in allowed: allowed.remove('←')
        # Force arrow toward end if adjacent.
        if abs(i - end[0]) + abs(j - end[1]) == 1:
            dx = end[0] - i
            dy = end[1] - j
            forced = None
            if dx == 1:
                forced = '↓'
            elif dx == -1:
                forced = '↑'
            if dy == 1:
                forced = '→'
            elif dy == -1:
                forced = '←'
            if forced in allowed:
                return [forced]
        if len(allowed) == 0:
            # Fallback: allow moves that keep within bounds.
            allowed = []
            if i > 0: allowed.append('↑')
            if i < n - 1: allowed.append('↓')
            if j > 0: allowed.append('←')
            if j < n - 1: allowed.append('→')
        return allowed

    # Helper: given an action, return possible next states as (row, col, probability)
    def get_next_states(i, j, action):
        if action == '↑':
            intended = (max(i-1, 0), j, 0.8)
            alt1 = (i, max(j-1, 0), 0.1)
            alt2 = (i, min(j+1, n-1), 0.1)
        elif action == '↓':
            intended = (min(i+1, n-1), j, 0.8)
            alt1 = (i, max(j-1, 0), 0.1)
            alt2 = (i, min(j+1, n-1), 0.1)
        elif action == '←':
            intended = (i, max(j-1, 0), 0.8)
            alt1 = (max(i-1, 0), j, 0.1)
            alt2 = (min(i+1, n-1), j, 0.1)
        elif action == '→':
            intended = (i, min(j+1, n-1), 0.8)
            alt1 = (max(i-1, 0), j, 0.1)
            alt2 = (min(i+1, n-1), j, 0.1)
        outcomes = [intended, alt1, alt2]
        combined = {}
        for o in outcomes:
            key = (o[0], o[1])
            combined[key] = combined.get(key, 0) + o[2]
        return [(key[0], key[1], prob) for key, prob in combined.items()]

    # Initialize value function and policy grids.
    V = [[0 for j in range(n)] for i in range(n)]
    policy_grid = [['' for j in range(n)] for i in range(n)]
    snapshots = []  # list of snapshots for each iteration

    threshold = 0.001
    maxDelta = float('inf')

    # Value iteration loop.
    while maxDelta > threshold:
        maxDelta = 0
        newV = [[0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                # Terminal states: end or obstacles.
                if grid[i][j] in ['end', 'obstacle']:
                    newV[i][j] = rewards[i][j]
                    policy_grid[i][j] = ''
                else:
                    allowed = get_allowed_directions(i, j)
                    bestActionValue = -float('inf')
                    bestAction = ''
                    for action in allowed:
                        outcomes = get_next_states(i, j, action)
                        expected_value = sum(prob * V[x][y] for (x, y, prob) in outcomes)
                        action_value = rewards[i][j] + gamma * expected_value
                        if action_value > bestActionValue:
                            bestActionValue = action_value
                            bestAction = action
                    newV[i][j] = bestActionValue
                    policy_grid[i][j] = bestAction
                delta = abs(newV[i][j] - V[i][j])
                if delta > maxDelta:
                    maxDelta = delta
        V = newV
        # Append a snapshot after each iteration.
        snapshots.append({'V': [row[:] for row in V],
                          'policy': [row[:] for row in policy_grid],
                          'converged': False})
    # Mark the final snapshot as converged.
    if snapshots:
        snapshots[-1]['converged'] = True

    return jsonify({'snapshots': snapshots})

if __name__ == '__main__':
    app.run(debug=True)
