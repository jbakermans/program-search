from API import *
from utilities import *

class ProgramGraph:
    """A program graph is a state in the search space"""
    def __init__(self, nodes):
        self.nodes = nodes if isinstance(nodes, tuple) else tuple(nodes)

    @staticmethod
    def fromRoots(rs, oneParent=True):
        assert oneParent
        if not oneParent:
            ns = set()
            def reachable(n):
                if n in ns: return
                ns.add(n)
                for c in n.children():
                    reachable(c)
            reachable(r)
            return ProgramGraph(ns)
        else:
            ns = []
            def visit(n):
                ns.append(n)
                for c in n.children(): visit(c)
            for r in rs: visit(r)
            return ProgramGraph(ns)
    @staticmethod
    def fromRoot(r,oneParent=True):
        return ProgramGraph.fromRoots([r],oneParent=oneParent)
                    

    def __len__(self): return len(self.nodes)

    def prettyPrint(self,letters=False):
        variableOfNode = [None for _ in self.nodes]
        nameOfNode = [None for _ in self.nodes] # pp of node

        lines = []

        def getIndex(p):
            for i, pp in enumerate(self.nodes):
                if p is pp: return i
            assert False                

        def pp(j):
            if variableOfNode[j] is not None: return variableOfNode[j]
            serialization = [t if not isinstance(t,Program) else pp(getIndex(t))
                             for t in self.nodes[j].serialize()]
            expression = f"({' '.join(map(str, serialization))})"
            if letters:
                variableOfNode[j] = f"{chr(ord('A') + len(lines))}"
            else:
                variableOfNode[j] = f"${len(lines)}"
            lines.append(f"{variableOfNode[j]} <- {expression}")
            return variableOfNode[j]

        for j in range(len(self.nodes)):
            pp(j)
        return "\n".join(lines)
                          
    def extend(self, newNode):
        return ProgramGraph([newNode] + list(self.nodes))

    def objects(self, oneParent=True):
        return [o for o in self.nodes
                if not oneParent or not any( any( c is o for c in op.children() ) for op in self.nodes )]
