import re



def remove_unwanted_text(papers) -> list : 
    
    for paper in papers : 
        if(len(paper) == 0 or paper[0].isdigit()==0) : 
            papers.remove(paper)
            
    return papers


def extract_related_papers(papers) -> list : 
    
    print("Responses received!!")
    
    papers = papers.split("\n")
    
    print("Removing unwanted text...")
    
    papers = remove_unwanted_text(papers)
    cleaned_list = [re.sub(r'^\d+\.\s*-', '', item) for item in papers]
    
    print("Removed unwanted text!!")
    
    return cleaned_list
    