import os
import tiktoken
from transformers.utils import logging
logging.set_verbosity_error()
from DynamiTree.utils import *
from DynamiTree.answer_prompt import ANSWER_QUERY

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.current_device()}")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU.")

token_counts = []
def count_tokens(text):
    """Estimate token count for a given text using tiktoken."""
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    return len(encoding.encode(text))


def get_last_session_file(cache_dir):
    """Retrieve the last session file based on session number."""
    files = [f for f in os.listdir(cache_dir) if f.startswith("tree_session_") and f.endswith(".pkl")]
    if not files:
        return None
    session_numbers = sorted(
        [int(f.split("_")[2].split(".")[0]) for f in files],
        reverse=True
    )
    last_session_file = f"tree_session_{session_numbers[0]}.pkl"
    return os.path.join(cache_dir, last_session_file)


TEMP_TREE_PATH = "results/temp_bptree"
os.makedirs(TEMP_TREE_PATH, exist_ok=True)

def initialize_tree_and_insert(fact: str, document_type="LFCO", max_degree=3):
    """
    Load an existing tree if available, else create one.
    Insert a new fact, generate updated summaries, and save.
    """
    tree_path = os.path.join(TEMP_TREE_PATH, "temp.pkl")

    if os.path.exists(tree_path):
        tree = load_tree(tree_path)
        print("Loaded existing tree.")
    else:
        tree = BPlusTree(max_degree=max_degree)
        print("Created new B+ Tree.")

    tree.insert(fact)
    print("Inserted new fact.")

    tree.generate_summaries(document_type=document_type)
    print("Updated summaries.")

    save_tree(tree, name="temp", cache_path=TEMP_TREE_PATH)
    tree.visualize(TEMP_TREE_PATH, "tree_visualization.txt")
    print("Tree saved & visualized.")



def query_tree_and_answer(query: str, model_name="gpt-4o-mini", top_k=5):
    """
    Load the saved tree, retrieve top-k facts and summaries, generate answer.
    """
    tree_path = os.path.join(TEMP_TREE_PATH, "temp.pkl")
    tree = load_tree(tree_path)
    if tree is None:
        print("Tree not found. Please run initialize_tree_and_insert() first.")
        return

    top_facts = tree.retrieve_top_k_facts(query, k=top_k)
    summaries = tree.retrieve_summaries(tree.root)
    summary_text = "\n".join(summaries) if summaries else "No summary available."

    context = "\n".join([fact for _, fact in top_facts]) + "\nSummaries: " + summary_text
    prompt = ANSWER_QUERY.format(contexts=context, question=query)

    print(f"Prompt tokens: {count_tokens(prompt)}")

    messages = [[{"role": "user", "content": prompt}]]
    response = openai_async_inference(messages, model_name=model_name)[0]

    print("\n Answer:")
    print(response.get("answer", "No answer returned."))


def reset_tree():
    """
    Remove the stored B+ Tree and its visualization.
    """
    tree_file = os.path.join(TEMP_TREE_PATH, "temp.pkl")
    vis_file = os.path.join(TEMP_TREE_PATH, "tree_visualization.txt")

    removed_any = False

    if os.path.exists(tree_file):
        os.remove(tree_file)
        print("Removed saved tree.")
        removed_any = True

    if os.path.exists(vis_file):
        os.remove(vis_file)
        print("Removed tree visualization.")
        removed_any = True

    if not removed_any:
        print("Nothing to clear.")



if __name__ == "__main__":
    reset_tree()

    initialize_tree_and_insert("Hillary Clinton, a prominent figure in American politics since her husband Bill Clinton\u2019s presidency, leaves her role as Secretary of State with nearly 70% approval ratings, higher than any outgoing secretary since Colin Powell. Despite being named the most admired woman 17 times by Gallup, she has faced significant criticism, particularly from conservative opponents who viewed her as polarizing. As First Lady, Clinton was politically active, attempting healthcare reform but facing backlash. Her run for the presidency in 2008 presented challenges; her campaign aimed to portray her as tough, but this approach sometimes backfired, making her seem cold. After a contentious primary battle, Barack Obama won the nomination but surprised many by appointing Clinton as Secretary of State. Despite any potential bitterness, she accepted, displaying professionalism and pragmatism. In her early weeks in the role, foreign leaders welcomed her warmly, acknowledging her global stature. David Miliband, then British Foreign Secretary, praised her as an ambassador not just for America, but for its ideals, illustrating the respect she garnered internationally throughout her career.")
    initialize_tree_and_insert("Despite Hillary Clinton's extensive efforts as Secretary of State, critics question the tangible outcomes of her four years in office. The \"reset\" with Russia has been deemed unsuccessful, and concerns about Iran's nuclear ambitions remain high. Clinton chose not to take risks in the complex realm of Middle East diplomacy but emphasized the importance of restoring American leadership and coalition-building, according to her deputy, Jake Sullivan. This broader legacy is framed as essential for navigating the modern geopolitical landscape with emerging powers like Brazil and Turkey. Throughout her tenure, Clinton's personal image evolved, revealing a more relaxed and approachable persona that resonated with the public. This transformation allowed her to move beyond her previous identity as \"Clinton number two,\" gaining recognition as Hillary Rodham Clinton. Observers noted her ability to shift from a divisive figure in American politics to one respected by some Republicans, showcasing her political acumen. However, her final months were overshadowed by the tragic attack in Benghazi, which ignited partisan conflict over her handling of security matters. While she emerged largely unscathed from congressional testimony, health issues reminded the public of her age. Speculation about a potential presidential run loomed, with supporters and friends expressing hope for her return to politics, despite her insistence on stepping back. Clinton has indicated that she remains open to opportunities while prioritizing personal time for rest and reflection before any potential decision.")
    initialize_tree_and_insert("Hillary Clinton's tenure as Secretary of State was marked by a complex balancing act between advancing US interests and upholding American values in international relations. Jake Sullivan, her deputy chief of staff, emphasized the importance of showcasing the pragmatism and maturity of the US-China relationship amidst tensions. However, repositioning the US for the 21st century proved challenging, particularly during the Arab Spring. Clinton initially misjudged the stability of the Egyptian government, stating it was stable just as protests began, leading to President Hosni Mubarak's ousting. Critics, including Eliot Abrams, argue that Clinton's responses in Libya and Syria were slow and ineffective, highlighting a perceived indecisiveness that resulted in tragic consequences. In Bahrain, the administration faced backlash for not supporting pro-democracy demonstrators against a brutal crackdown, prioritizing strategic interests over human rights due to Bahrain's significance as a base for the US Navy. Despite frustrations over the administration's cautious approach, particularly regarding Syria, Clinton advocated for multilateral diplomacy. Turkey's Foreign Minister praised her collaborative approach, recognizing that while representing a global power may seem straightforward, it involves navigating the complexities of international relationships without imposing American values.")
    initialize_tree_and_insert("Four years into her role as Secretary of State, Hillary Clinton was viewed as a diplomatic \"rock star,\" as noted by David Miliband, reflecting her significant global presence. When appointed by President Obama, she was tasked with restoring America\u2019s tarnished reputation following the Bush administration. Clinton embarked on a campaign to improve perceptions of the US, even in countries like Pakistan, where she faced significant hostility. During her visits, she engaged directly with locals, demonstrating warmth and empathy despite criticism. Her efforts included addressing tensions caused by US drone strikes and other military actions. Clinton\u2019s pragmatic approach helped navigate a crisis after a NATO strike killed Pakistani soldiers, leading to a carefully worded apology from Washington. She built strong relationships with world leaders, which proved crucial for US diplomacy. Her extensive travel and energy set her apart, allowing her to redefine American foreign policy through a \"smart power\" approach, balancing diplomatic, military, and cultural tools. Clinton prioritized women's rights and development issues while fostering collaboration between the State Department and the Pentagon. Despite facing criticism over human rights discussions, she maintained a comprehensive approach, exemplified by her handling of a diplomatic crisis involving Chinese dissident Chen Guangcheng in 2012.")
    query_tree_and_answer("How did Hillary Clinton's approval ratings compare to those of her predecessors as Secretary of State?")


