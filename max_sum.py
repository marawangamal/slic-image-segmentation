import numpy as np

class Node:
    def __init__(self, unary, index, value):
        """ Initializes a single node
        Args:
            unary: (np.array) unary pottentials for K states, Shape: [K,1]
            index: (integer) node index, should be unique for each node in a graph
        """
        self.out_msgs = []
        self.K = len(unary)
        self.unary = unary
        self.neighbors = [None]*4 # [left, up, right, down]
        self.idx = index
        self.value = value
        self.num_neighbors = 4


        self.in_msgs = []

        for i in range(self.num_neighbors):
            self.in_msgs.append( np.copy(np.ones((self.K,1)))  * 0.0)

class LBP:
    """ Loopy Belief Propagation Engine. Pass in a list of graph as a list of Node Class objects and then run
    Eg.
        lbp = LBP(nodes)
        proba, labels = lbp.run()
    """
    def __init__(self, nodes, num_classes):
        """ Initializes graph for loopy belief propagation
        Args:
            nodes: (python list) list of Node class objects
        """
        # self.img = img
        self.dirs = ['left', 'up', 'right', 'down']   # Schedule
        self.dir_node_dict = {'left':'right', 'up':'down', 'right':'left', 'down':'up', }
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.C = num_classes

    def set_msg(self, node, dir, msg):
        """ sets message outgoing from node in 'dir' direction
        Args:
            dir: (string) direction from self.dirs
            node: (Node class) node from which message originates
            msg: (np.array) msg from Node to 'dir' neighbor, Shape: [K,1]
        """

        node_neighbor_i  = self.dirs.index(dir)                      # curr node's neighbor index
        neighbor_node_i = self.dirs.index(self.dir_node_dict[dir])    # neighbor's index of curr node
        node.neighbors[node_neighbor_i].in_msgs[neighbor_node_i] = msg

    def pairwise_pott(self, n1, n2):
        """ Computes pairwise pottential matrix between nodes, entries are cost of an assinment.
            Phi(X1, X2) = exp( - ||I1 - I2|| / sigma) if L1 =/= L2
        Args:
            nx: (Node Class) node in graph
        Returns:
            phi: (np.array) pairwise pottentials/costs since using gibbs dist., Shape: [K, K] where K is the number of states of node
        """

        # Smoothness to Conv prediction importance ratio (hyperparameter)
        w1 = 1

        F1 = n1.value
        F2 = n2.value

        I1 = np.copy(F1[:3]).astype(np.float32) / 255.0
        I2 =  np.copy(F2[:3]).astype(np.float32) / 255.0

        sigma1 = 1

        phi =  np.ones((self.C,self.C)) * w1 * np.exp( - np.linalg.norm(I1-I2) / sigma1 )
        np.fill_diagonal(phi, 0)


        return phi


    def update(self, dir, node):
        """ Sends message from a node to its 'dir' neighbor MU_x1->x2(x2) = ln
        Args:
            dir: (string) direction from self.dirs
            node: (Node class) node from which message originates
        """

        ni = self.dirs.index(dir)                   # neighbor index
        except_mu = node.in_msgs[ni]                # excluded msg


        sum_mu_neighbors = np.sum(np.asarray(node.in_msgs), axis = 0) - except_mu  #[5,1] - [5,1]

        if(node.neighbors[ni] != None):
            # Costs
            phi_pairwise = self.pairwise_pott(node, node.neighbors[ni]) # [K, K] + [1, K] ]
            phi_unary = node.unary.max() - node.unary

            # Factors
            f_pairwise = np.exp(- phi_pairwise)
            f_unary = np.exp(- phi_unary)

            # Using ln(f) instead for max-sum variant
            f_pairwise = - phi_pairwise
            f_unary = - phi_unary

            msg2neighbor = f_pairwise + np.transpose(f_unary) + np.transpose(sum_mu_neighbors)
            msg2neighbor = np.expand_dims(np.amax(msg2neighbor, axis=0), axis = 1)

            # Normalize
            msg2neighbor_normalized = msg2neighbor / np.abs(np.sum(msg2neighbor))
            self.set_msg(node, dir, msg2neighbor_normalized)


    def run(self, iters=100):
        """ Runs loopy belief propagation on graph
        Returns:
            node_labels: (np.array) inferred node label indices. Shape: [N,]
            probs: (np.array) probabilty of labelling at each iteration. Shape: [iters,]
        """

        probs = np.zeros(iters)
        energys = np.zeros(iters)
        node_labels = np.ones(self.num_nodes) * -1

        for i in range(iters):
            for n in range(self.num_nodes):
                self.update('left', self.nodes[n])
                self.update('up', self.nodes[n])
                self.update('right', self.nodes[n])
                self.update('down', self.nodes[n])

            PX = 1

            e = 0


            for j, node in enumerate(self.nodes):

                logPx_unnormalized = np.sum(np.asarray(node.in_msgs), axis = 0)
                node_labels[j] = np.argmax(logPx_unnormalized, axis = 0)
                e += np.amax(logPx_unnormalized, axis = 0)

                Px_unnormalized = np.exp(logPx_unnormalized)
                Z = np.sum(Px_unnormalized)

                # Px_normalized = Px_unnormalized/Z
                # PX *= np.amax(Px_normalized)

            energys[i] = e

        return energys, node_labels

class graph_builder:
    """ Builds graph for image super-pixels
    """
    def __init__(self, sets, mus, unarys):
        """
        Args:
            sets: (np.array) superpixel membership integer mask, Shape: [rows, cols, 1]
            mus: (np.array) sueprpixel centers and colour, Shape: [K, 5]
            unarys: (np.array) unary pottentials, Shape: [rows, cols, 3]
        """
        self.unarys = unarys
        self.mus = mus
        self.sets = sets

        self.K, _ = mus.shape
        rows, cols, self.C = self.unarys.shape
        self.S = int(np.sqrt((cols*rows)/self.K))

        self.nodes = []

        # Set unary values for ever cluster center (Conv Net)
        self.set_unarys()

        for k in range(self.K):
            self.nodes.append(Node(self.mu_unarys[k], k, self.mus[k]))

    def set_unarys(self):
        """ Sets the unary pottentials for sueprpixels (cluster centers). Simply takes unary of cluster center
        """

        self.mu_unarys = np.zeros((self.K, self.C))

        for i, mu in enumerate(self.mus):
            row, col = self.mus[i, 3:]
            self.mu_unarys[i] = self.unarys[row, col]

    def get_nodes(self):
        """ Creates nodes from superpixels, infers neighbors of each superpixel
        Returns:
            nodes: (list) of Node class objects
        """

        mus1 = np.expand_dims(np.copy(self.mus[:,3:]), 0) # [1, K, 2]
        mus2 = np.moveaxis(np.expand_dims(np.copy(self.mus[:,3:]), 0), [0,1,2], [1,0,2]) # [K, 1, 2]
        # mus2 = np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape


        # Shape: [K,K,2]
        dists = (np.abs(mus1 - mus2)).astype(np.float32)

        h_neighbors = np.zeros((2, 2))
        v_neighbors = np.zeros((2, 2))


        for i in range(self.K):

            # horizontal and vertical distances to cluster i
            h_dists = np.copy(dists[i, :, 1]) + 1   # [K, 1] row2-row1, col2-col1
            v_dists = np.copy(dists[i, :, 0]) + 1   # [K, 1]

            h_dists[i] = np.inf
            v_dists[i] = np.inf

            # keeps only clusters within same row (vertical)
            h_dists_within = (np.abs(v_dists) < self.S/2.0).astype(np.float32)
            h_dists_within[h_dists_within == 0] = np.inf

            # keeps only clusters on within same column (horizontal)
            v_dists_within = (np.abs(h_dists) < self.S/2.0).astype(np.float32)
            v_dists_within[v_dists_within == 0] = np.inf

            # sets dists not on same level to inf
            h_dists = h_dists * h_dists_within
            v_dists = v_dists * v_dists_within

            # Left and Right Neighbors

            h_neighbor_indices = np.argpartition(h_dists, 2)[:2]                    # indices of k closest horizontal dists
            hn1, hn2 = h_neighbor_indices[0], h_neighbor_indices[1]                 # smallest two horizontal dists
            # h_neighbors[0], h_neighbors[1] = self.mus[vn1, 1], self.mus[vn2, 1]

            ln_i1 = np.argmin(  np.array([self.mus[hn1, 4], self.mus[hn2, 4]])  )                   # left horizontal neighbor will have smaller col value
            ln_index = h_neighbor_indices[ln_i1]                                    # cluster index

            rn_i1= np.argmax(  np.array([ self.mus[hn1, 4], self.mus[hn2, 4] ])  )
            rn_index = h_neighbor_indices[rn_i1]

            # Up and Down neighbors

            v_neighbor_indices = np.argpartition(v_dists, 2)[:2]
            vn1, vn2 = v_neighbor_indices[0], v_neighbor_indices[1]

            un_i1 = np.argmin(   np.array([self.mus[vn1, 3], self.mus[vn2, 3]])   )                   # up vertical neighbor will have smaller row value
            un_index = v_neighbor_indices[un_i1]
            dn_i1 = np.argmax(   np.array([self.mus[vn1, 3], self.mus[vn2, 3]])   )
            dn_index = v_neighbor_indices[dn_i1]

            # Assign Neighbors

            leftNode = self.nodes[ln_index]
            upNode = self.nodes[un_index]
            rightNode = self.nodes[rn_index]
            downNode = self.nodes[dn_index]

            self.nodes[i].neighbors = [leftNode, upNode, rightNode, downNode]


        return self.nodes
