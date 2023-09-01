/**
 * RL Game: Neural Collector (DQN Implementation)
 * Training happens at the END of each episode, not every step.
 */

class GameEnvironment {
    constructor(gridSize = 4) {
        this.gridSize = gridSize;
        this.reset();
    }

    reset() {
        this.agentPos = { x: 0, y: 0 };
        this.goalPos = { x: this.gridSize - 1, y: this.gridSize - 1 };
        this.steps = 0;
        this.maxSteps = 40; // 更短的回合，更快收敛
        return this.getState();
    }

    get obstacles() {
        // 4x4 网格下的障碍，包括指定的 (0,3)
        return [[0, 3], [1, 1], [1, 2], [2, 1]].map(p => ({ x: p[0], y: p[1] }));
    }

    isObstacle(x, y) {
        return this.obstacles.some(o => o.x === x && o.y === y);
    }

    getState() {
        return [
            this.agentPos.x / (this.gridSize - 1),
            this.agentPos.y / (this.gridSize - 1),
            this.goalPos.x / (this.gridSize - 1),
            this.goalPos.y / (this.gridSize - 1)
        ];
    }

    step(action) {
        const prevDist = Math.abs(this.agentPos.x - this.goalPos.x) + Math.abs(this.agentPos.y - this.goalPos.y);
        let nx = this.agentPos.x;
        let ny = this.agentPos.y;

        if (action === 0 && ny > 0) ny--;
        else if (action === 1 && ny < this.gridSize - 1) ny++;
        else if (action === 2 && nx > 0) nx--;
        else if (action === 3 && nx < this.gridSize - 1) nx++;

        this.steps++;

        let reward = -0.05;
        let done = false;

        if (this.isObstacle(nx, ny)) {
            reward = -0.8; // Penalty but don't move
        } else {
            this.agentPos.x = nx;
            this.agentPos.y = ny;
            const newDist = Math.abs(nx - this.goalPos.x) + Math.abs(ny - this.goalPos.y);
            if (nx === this.goalPos.x && ny === this.goalPos.y) {
                reward = 10.0;
                done = true;
            } else if (newDist < prevDist) {
                reward = 0.2;
            }
        }

        if (this.steps >= this.maxSteps) done = true;

        return { state: this.getState(), reward, done };
    }
}

class DQNAgent {
    constructor() {
        this.model = this.buildModel();
        this.targetModel = this.buildModel();
        this.memory = [];
        this.maxMemory = 3000;
        this.gamma = 0.95;
        this.epsilon = 1.0;
        this.epsilonDecay = 0.97; // 更快衰减，更快开始利用学到的知识
        this.epsilonMin = 0.05;
        this.batchSize = 64;
        this.trainCount = 0;
    }

    buildModel() {
        const m = tf.sequential();
        m.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [4] }));
        m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        m.add(tf.layers.dense({ units: 4, activation: 'linear' }));
        m.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
        return m;
    }

    act(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 4);
        }
        return tf.tidy(() => {
            const q = this.model.predict(tf.tensor2d([state]));
            return q.argMax(1).dataSync()[0];
        });
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push({ state, action, reward, nextState, done });
        if (this.memory.length > this.maxMemory) this.memory.shift();
    }

    async train() {
        if (this.memory.length < this.batchSize) return;

        // 每次回合结束训练 4 个 Batch，加速收敛
        for (let t = 0; t < 4; t++) {
            const batch = [];
            for (let i = 0; i < this.batchSize; i++) {
                batch.push(this.memory[Math.floor(Math.random() * this.memory.length)]);
            }

            const statesData = batch.map(b => b.state);
            const nextStatesData = batch.map(b => b.nextState);

            // Compute targets inside tf.tidy to prevent leaks
            const targets = tf.tidy(() => {
                const currentQs = this.model.predict(tf.tensor2d(statesData)).arraySync();
                const nextQs = this.targetModel.predict(tf.tensor2d(nextStatesData)).max(1).dataSync();
                batch.forEach((b, i) => {
                    currentQs[i][b.action] = b.done
                        ? b.reward
                        : b.reward + this.gamma * nextQs[i];
                });
                return tf.tensor2d(currentQs);
            });

            const statesTensor = tf.tensor2d(statesData);
            await this.model.fit(statesTensor, targets, { epochs: 1, verbose: 0 });

            statesTensor.dispose();
            targets.dispose();
        }

        if (this.epsilon > this.epsilonMin) this.epsilon *= this.epsilonDecay;
        this.trainCount++;
        if (this.trainCount % 5 === 0) {
            this.targetModel.setWeights(this.model.getWeights());
        }
    }
}

class Visualizer {
    constructor(canvasId, gridSize) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.gridSize = gridSize;
        this.cellSize = this.canvas.width / this.gridSize;
    }

    draw(env, agent, insightMode) {
        const ctx = this.ctx;
        const cs = this.cellSize;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (insightMode) this.drawValueMap(agent, env.goalPos);

        // Grid
        ctx.strokeStyle = '#e8e8e8';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= this.gridSize; i++) {
            ctx.beginPath(); ctx.moveTo(i * cs, 0); ctx.lineTo(i * cs, this.canvas.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, i * cs); ctx.lineTo(this.canvas.width, i * cs); ctx.stroke();
        }

        // Obstacles
        ctx.fillStyle = '#c0c0c0';
        env.obstacles.forEach(o => {
            ctx.fillRect(o.x * cs + 3, o.y * cs + 3, cs - 6, cs - 6);
        });

        // Goal
        ctx.fillStyle = '#d13030';
        ctx.fillRect(env.goalPos.x * cs + 5, env.goalPos.y * cs + 5, cs - 10, cs - 10);

        // Agent
        ctx.fillStyle = '#1a1a1a';
        ctx.beginPath();
        ctx.arc(env.agentPos.x * cs + cs / 2, env.agentPos.y * cs + cs / 2, cs / 3, 0, Math.PI * 2);
        ctx.fill();
    }

    drawValueMap(agent, goalPos) {
        const g = this.gridSize;
        const cs = this.cellSize;
        tf.tidy(() => {
            const states = [];
            for (let x = 0; x < g; x++)
                for (let y = 0; y < g; y++)
                    states.push([x / (g - 1), y / (g - 1), goalPos.x / (g - 1), goalPos.y / (g - 1)]);
            const qVals = Array.from(agent.model.predict(tf.tensor2d(states)).max(1).dataSync());

            // 修正后的动态归一化：
            // 找出全场的最小和最大 Q 值 
            const minQ = Math.min(...qVals);
            const maxQ = Math.max(...qVals);
            let range = maxQ - minQ;
            if (range < 0.001) range = 1; // 防止一开始全一样时除以0报错或显示全红

            let idx = 0;
            for (let x = 0; x < g; x++) {
                for (let y = 0; y < g; y++) {
                    const q = qVals[idx++];
                    // 只有当前的 Q 值大于 -1 (碰撞障碍物的惩罚是 -1，移动惩罚 -0.05) 才需要明显显示
                    let intensity = 0;
                    if (q > minQ) { // 如果不是最差的那一批
                        intensity = (q - minQ) / range;
                    }

                    // 稍微放大视觉效果但绝不超过 0.8 透明度，免得看不清格子里的东西
                    const alpha = Math.min(0.8, intensity * 0.7);

                    this.ctx.fillStyle = `rgba(37, 99, 235, ${alpha})`;
                    this.ctx.fillRect(x * cs, y * cs, cs, cs);
                }
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const env = new GameEnvironment();
    const agent = new DQNAgent();
    const viz = new Visualizer('rl-canvas', env.gridSize);

    let episode = 0;
    let isRunning = false;
    let insightMode = false;

    const epDisplay = document.getElementById('ep-count');
    const scoreDisplay = document.getElementById('cur-score');
    const overlay = document.getElementById('game-overlay');
    const insightToggle = document.getElementById('insight-toggle');

    insightToggle.addEventListener('change', e => insightMode = e.target.checked);

    document.getElementById('start-btn').addEventListener('click', () => {
        if (isRunning) return;
        overlay.style.display = 'none';
        isRunning = true;
        runEpisode();
    });

    // Run one episode at a time using requestAnimationFrame for smooth animation
    function runEpisode() {
        let state = env.reset();
        let score = 0;
        let done = false;

        function step() {
            if (!isRunning) return;

            const action = agent.act(state);
            const result = env.step(action);
            agent.remember(state, action, result.reward, result.state, result.done);
            state = result.state;
            score += result.reward;
            done = result.done;

            viz.draw(env, agent, insightMode);
            scoreDisplay.innerText = score.toFixed(1);

            if (!done) {
                // Use setTimeout for throttled animation (50ms per step → ~20fps)
                setTimeout(step, 50);
            } else {
                episode++;
                epDisplay.innerText = episode;
                // Train at end of each episode, then start next
                agent.train().then(() => {
                    setTimeout(runEpisode, 10);
                });
            }
        }

        step();
    }
});
