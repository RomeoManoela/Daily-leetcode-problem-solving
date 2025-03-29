# Définition de la classe pour un nœud d'arbre binaire
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# Fonction de parcours en pré-ordre
def pre_order(root):
    if root:
        print(root.value, end=" ")  # Visiter le nœud courant
        pre_order(root.left)  # Parcours du sous-arbre gauche
        pre_order(root.right)  # Parcours du sous-arbre droit


# Fonction de parcours en in-ordre
def in_order(root):
    if root:
        in_order(root.left)  # Parcours du sous-arbre gauche
        print(root.value, end=" ")  # Visiter le nœud courant
        in_order(root.right)  # Parcours du sous-arbre droit


# Fonction de parcours en post-ordre
def post_order(root):
    if root:
        post_order(root.left)  # Parcours du sous-arbre gauche
        post_order(root.right)  # Parcours du sous-arbre droit
        print(root.value, end=" ")  # Visiter le nœud courant


# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un arbre binaire
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)

    print("Parcours en pré-ordre :")
    pre_order(root)  # Affiche 1 2 4 5 3
    print("\nParcours en in-ordre :")
    in_order(root)  # Affiche 4 2 5 1 3
    print("\nParcours en post-ordre :")
    post_order(root)  # Affiche 4 5 2 3 1
