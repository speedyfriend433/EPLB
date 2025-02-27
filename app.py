import torch
from flask import Flask, render_template, request, jsonify
import eplb
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    weight_data = data.get('weights', [])
    num_replicas = int(data.get('num_replicas', 16))
    num_groups = int(data.get('num_groups', 4))
    num_nodes = int(data.get('num_nodes', 2))
    num_gpus = int(data.get('num_gpus', 8))
    weight = torch.tensor(weight_data, dtype=torch.float)
    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    
    result = {
        'phy2log': phy2log.tolist(),
        'log2phy': log2phy.tolist(),
        'logcnt': logcnt.tolist(),
        'num_nodes': num_nodes,
        'num_gpus': num_gpus,
        'gpus_per_node': num_gpus // num_nodes
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)