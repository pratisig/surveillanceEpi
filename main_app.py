# ============================================================
# ROUTAGE AVEC DÉBOGAGE
# ============================================================

# Debug : afficher la page actuelle
# st.sidebar.info(f"Page actuelle : {st.session_state.page_choice}")

if st.session_state.page_choice == "Paludisme":
    try:
        import app_paludisme
    except Exception as e:
        st.error(f"Erreur de chargement de l'application Paludisme : {e}")
        st.code(str(e))
        # Retour à l'accueil en cas d'erreur
        st.session_state.page_choice = "Accueil"
        if st.button("Retour à l'accueil"):
            st.rerun()
    
elif st.session_state.page_choice == "Rougeole":
    try:
        import app_rougeole
    except Exception as e:
        st.error(f"Erreur de chargement de l'application Rougeole : {e}")
        st.code(str(e))
        st.session_state.page_choice = "Accueil"
        if st.button("Retour à l'accueil"):
            st.rerun()
    
elif st.session_state.page_choice == "Manuel":
    try:
        import app_manuel
    except Exception as e:
        st.error(f"Erreur de chargement du Manuel : {e}")
        st.code(str(e))
        st.session_state.page_choice = "Accueil"
        if st.button("Retour à l'accueil"):
            st.rerun()

elif st.session_state.page_choice == "Accueil":
    # ... TOUT LE CODE DE LA PAGE D'ACCUEIL ...
