def retrieve_relevant_context(collection, query, n_results=5, source_filter=None):
    """
    Retrieves the most similar content from ChromaDB collection based on user query.
    source_filter: optional list or string filter for metadata 'source_type' (e.g., 'doc', 'screen_crop')
    """
    where_clause = {}
    if source_filter:
        if isinstance(source_filter, list):
            if len(source_filter) > 1:
                where_clause = {"source_type": {"$in": source_filter}}
            elif len(source_filter) == 1:
                where_clause = {"source_type": source_filter[0]}
        else:
            where_clause = {"source_type": source_filter}
            
    # Perform query
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Parse and format results
        formatted_docs = []
        if results and 'documents' in results and results['documents']:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else [{}] * len(docs)
            distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metadatas, distances):
                formatted_docs.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
        return formatted_docs
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

if __name__ == "__main__":
    print("Retriever module loaded.")
