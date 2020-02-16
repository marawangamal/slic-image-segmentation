class Node:
    def __init__(self, unary):
        """ Initializes a single node
        Args:
            unary: (np.array) unary pottentials for K states, Shape: [K,1]
        """
        self.out_msgs = []
        self.K = len(unary)
        self.unary = np.zeros((self.K,1))
        self.neighbors = [None]*4

        self.in_msgs = []

        for i in range(self.K):
            self.in_msgs.append( np.copy(np.ones((2,1))) )

class LBP:
    def __init__(self, nodes):
        """ Initializes graph for loopy belief propagation
        Args:
            nodes: (python list) list of Node class objects
        """
        self.img = img
        self.dirs = ['left', 'up', 'right', 'down']   # Schedule
        self.dir_node_dict = {'left':'right', 'up':'down', 'right':'left', 'down':'up', }
        self.nodes = nodes

    def set_msg(self, node, dir, msg):
        """ sets message outgoing from node in dir direction
        Args:
            dir: (string) direction from self.dirs
            node: (Node class) node from which message originates
            msg: (np.array) msg from Node to 'dir' neighbor, Shape: [K,1]
        """
        ni  = self.dirs.index(dir)                      # curr node's neighbor index
        j = self.dirs.index(self.dir_node_dict[dir])    # neighbor's index of curr node
        node.neighbors[ni].in_msgs[j] = msg

    def pairwise_pott(self, coords1, coords2):
        """ Computes pairwise pottential mat between nodes centered at coords1, coords2
        Args:
            coordsx: (int tuple) coordinates in form [row, col]
        Returns:
            pott_arr: (np.array) pairwise pottentials, Shape: [K, K] where K is the number of states of node
        """
        pass

    def update(self, dir, node):
        """ Sends a single msg from a node to its 'dir' neighbor
        Args:
            dir: (string) direction from self.dirs
            node: (Node class) node from which message originates
        """

        ni = self.dirs.index(dir)               # neighbor index
        except_mu = node.in_msgs[i]             # excluded msg

        if(node.neighbors[i] != None):
            # [K, K] + [1, K]
            msg2neighbor = self.pairwise_pott(node.coords, node.neighbors.coords) + np.transpose(node.unary)
            msg2neighbor = np.amax(msg2neighbor, axis=0)
            self.set_msg(node, dir, msg)

    def run(self, iters=10):
        """ Runs loopy belief propagation on graph
        """

        probs = np.zeros(iters)
        node_labels = np.ones(N) * -1

        for i in range(inters):
            for node in self.nodes:
                self.update('left')
                self.update('up')
                self.update('right')
                self.update('down')

            Px = 1

            for i, node in enumerate(self.nodes):
                argmax_x = np.sum(np.asarray(node.in_msgs), axis = 0)
                node_labels[i] = argmax_x
                max_x = np.amax( np.sum(np.asarray(node.in_msgs), axis = 0)  )
                Px *= pmax_x

            probs[i] = PX


        return probs


class graph_builder:
    """ Builds graph for image super-pixels
    """
    def __init__(self, sets, mus, unarys):
        """
        Args:
            sets: (np.array) superpixel membership integer mask, Shape: [Rows, Cols, 1]
            mus: (np.array) sueprpixel centers and colours
        """
        self.unarys = unarys
        self.mus = mus
        self.sets = sets

    def create_nodes(self):
        """ Creates nodes from superpixels, infers neighbors of each superpixel
        Returns:
            nodes: (list) of Node class objects
        """

        mus1 = np.copy(self.mus) # [1, K, 2]
        mus2 = np.copy(self.mus) # [K, 1, 2]

        nodes = []

        # Shape: [K,K,2]
        dists = mus1 - mus2

        h_neighbors = np.zeros((2, 2))
        v_neighbors = np.zeros((2, 2))


        for i in range(K):
            h_dists= np.copy(dists[i, 0, :])  # [K, 1]
            v_dists= np.copy(dists[i, :, 0])  # [K, 1]

            h_dists[i] = np.inf
            v_dists[i] = np.inf

            h_neighbor_indices = np.argpartition(h_dists, 2)                    # indices of closest dists
            hn1, hn2 = h_neighbor_indices[0], h_neighbor_indices[1]             # indices of smallest two dist, correspond to kth center
            hn1_coords[0], hn1_coords[1] = self.mus[hn1], self.mus[hn2]         # coords of smallest two dists (ydir)

            ln_index = np.argmin(self.mus[hn1, 1], self.mus[hn2, 1])            # left neighbor will have smaller col value



            ln = h_neighbors[ln_index]
            rn_index = np.argmax(self.mus[hn1, 1], self.mus[hn2, 1])
            rn = h_neighbors[rn_index]

            v_neighbor_indices = np.argpartition(v_dists, 2)
            vn1, vn2 = v_neighbor_indices[0], v_neighbor_indices[1]
            v_neighbors[0], v_neighbors[1] = self.mus[vn1, 1], self.mus[vn2, 1]

            un_index = np.argmin(self.mus[vn1, 1], self.mus[vn2, 1])
            un = v_neighbors[un_index]
            dn_index = np.argmax(self.mus[vn1, 1], self.mus[vn2, 1])
            dn = v_neighbors[dn_index]


            node = Node(self.unarys[i])
            upNode = Node(self.unarys[upi])
            leftNode = Node(self.unarys[lefti])
            rightNode = Node(self.unarys[righti])
            downNode = Node(self.unarys[downi])

            node.neighbors = [leftNode, upNode, rightNode, downNode]
            nodes.append(node)

        return nodes
