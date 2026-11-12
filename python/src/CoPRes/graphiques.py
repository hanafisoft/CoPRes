"""Graphiques de CoPRes (matplotlib).

Trois couches :
- calcul (copres, fiche, stabilite, ...) : stable ;
- données du graphique (donnees_*) : un tableau (dictionnaire de colonnes) prêt à tracer, stable ;
- dessin (graph_*) : une proposition de mise en forme, EXPÉRIMENTALE, susceptible d'évoluer au fil
  des applications ; chaque fonction rend une figure matplotlib indépendante.

Langue des textes : `langue("fr")` ou `langue("en")` pour la session, ou l'argument `langue=` de
chaque fonction ; les textes sont dans donnees/textes.csv, fichier commun aux versions R et Python.
Thème : `theme(commune=..., taille=...)` modifie couleurs et taille du texte pour la session.

Conventions de couleur (palette contrôlée pour les daltonismes) : bleu, bloc porteur d'une
composante commune ou partagée ; vert, bloc porteur d'une composante propre ; gris clair, bloc non
porteur ; orange, mélange ; gris moyen, part résiduelle. Groupes d'individus : ordre fixe.
"""
import os
import numpy as np

# ----------------------------------------------------------------------------- langue et thème
_TEXTES = None
_LANGUE = {"valeur": "fr"}
_THEME_DEFAUT = dict(commune="#2a78d6", propre="#199e70", non_porteur="#d9d8d4", melange="#eb6834",
                     residuel="#8a8984", encre="#0b0b0b", encre2="#52514e", grille="#e6e5e1",
                     groupes=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
                     seq=["#9ec5f4", "#5598e7", "#1c5cab"], taille=9)
_THEME = dict(_THEME_DEFAUT)


def langue(valeur=None):
    """Fixe ("fr" ou "en") ou renvoie la langue des graphiques pour la session."""
    if valeur is not None:
        if valeur not in ("fr", "en"):
            raise ValueError("langue : 'fr' ou 'en'")
        _LANGUE["valeur"] = valeur
    return _LANGUE["valeur"]


def theme(**reglages):
    """Modifie le thème graphique (clés : commune, propre, non_porteur, melange, residuel, encre,
    encre2, grille, groupes, seq, taille) ; theme(defaut=True) rétablit le thème d'origine.
    Renvoie le thème courant."""
    if reglages.pop("defaut", False):
        _THEME.clear()
        _THEME.update(_THEME_DEFAUT)
    for k, v in reglages.items():
        if k not in _THEME:
            raise ValueError(f"clé de thème inconnue : {k}")
        _THEME[k] = v
    return dict(_THEME)


theme_copres = theme      # même nom que dans la version R


def _t(cle, lg=None, **kw):
    global _TEXTES
    if _TEXTES is None:
        f = os.path.join(os.path.dirname(__file__), "donnees", "textes.csv")
        with open(f, encoding="utf-8") as fh:
            lignes = [l.rstrip("\n").split(";") for l in fh if l.strip()]
        _TEXTES = {l[0]: {"fr": l[1], "en": l[2]} for l in lignes[1:]}
    s = _TEXTES[cle][lg or _LANGUE["valeur"]]
    return s.format(**kw) if kw else s


def _nombre(x, n, lg):
    s = f"{x:.{n}f}"
    return s.replace(".", ",") if (lg or _LANGUE["valeur"]) == "fr" else s


def _netiq(L, noms, lg):
    if not L:
        return _t("residuelle", lg)
    return "{" + ",".join(str(noms[k]) if noms is not None else str(k + 1) for k in L) + "}"


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    T = _THEME
    matplotlib.rcParams.update({"axes.edgecolor": T["encre2"], "axes.labelcolor": T["encre"],
                                "xtick.color": T["encre2"], "ytick.color": T["encre2"],
                                "axes.spines.top": False, "axes.spines.right": False, "font.size": T["taille"]})
    return plt


def _noms(obj, noms=None):
    if noms is not None:
        return list(noms)
    if getattr(obj, "groupes", None) is not None:
        return list(obj.groupes)
    return [str(k + 1) for k in range(np.asarray(obj.saliences).shape[0])]


def _type(L):
    return "residuelle" if len(L) == 0 else ("propre" if len(L) == 1 else "commune")


def _couleur_type(t):
    return {"residuelle": _THEME["residuel"], "propre": _THEME["propre"], "commune": _THEME["commune"]}[t]


def _parties(obj):
    d = {}
    for h, L in enumerate(obj.etiquettes):
        d.setdefault(L, []).append(h)
    return d


# ============================================================================= données des graphiques
def donnees_saliences(obj, noms=None, langue=None):
    """Une ligne par bloc (ou groupe) et par composante : salience, seuil, porteur, type."""
    S, th, nm = np.asarray(obj.saliences), np.asarray(obj.seuils), _noms(obj, noms)
    out = {k: [] for k in ("composante", "etiquette", "bloc", "salience", "seuil", "porteur", "type")}
    for h, L in enumerate(obj.etiquettes):
        for k in range(S.shape[0]):
            out["composante"].append(h + 1)
            out["etiquette"].append(_netiq(L, nm, langue))
            out["bloc"].append(nm[k])
            out["salience"].append(float(S[k, h]))
            out["seuil"].append(float(th[k]))
            out["porteur"].append(k in L)
            out["type"].append(_type(L))
    return out


def donnees_coherence(obj, plan=(0, 1), noms=None, nb_points=361):
    """Pour chaque bloc : valeurs propres de la restriction au plan, seuil, état (porte tout,
    rien, une partie) et contours de l'ellipse et du cercle-seuil."""
    res, nm = obj.lode, _noms(obj, noms)
    Q = res.base[:, list(plan)]
    t = np.linspace(0, 2 * np.pi, nb_points)
    C = np.vstack([np.cos(t), np.sin(t)])
    out = []
    for k, a in enumerate(res.matrices):
        w, z = np.linalg.eigh(Q.T @ a @ Q)
        E = z @ np.diag(np.sqrt(np.clip(w, 0, None))) @ z.T @ C
        th = float(obj.seuils[k])
        etat = "porte_tout" if w[0] >= th else ("porte_rien" if w[-1] <= th else "porte_partie")
        out.append(dict(bloc=nm[k], lambda_min=float(w[0]), lambda_max=float(w[-1]), seuil=th, etat=etat,
                        ellipse_x=E[0], ellipse_y=E[1], cercle_x=np.sqrt(th) * C[0], cercle_y=np.sqrt(th) * C[1]))
    return out


def donnees_fiche(fiche, noms=None, langue=None):
    """Une ligne par composante de la méthode et par étiquette CoPRes : part, rôle (principale,
    mélange, résiduelle), plus mu* et r."""
    out = {k: [] for k in ("composante", "etiquette", "part", "role", "type", "mu", "r", "melangee", "diluee")}
    for h, c in enumerate(fiche.composition):
        Lp = fiche.principale[h]
        ordre = ([Lp] if Lp is not None else []) + [L for L in c if L and L != Lp] + ([()] if () in c else [])
        for L in ordre:
            out["composante"].append(h + 1)
            out["etiquette"].append(_netiq(L, noms, langue))
            out["part"].append(float(c.get(L, 0.0)))
            out["role"].append("principale" if L == Lp else ("residuelle" if L == () else "melange"))
            out["type"].append(_type(L))
            out["mu"].append(float(fiche.melange[h]))
            out["r"].append(float(fiche.residuel[h]))
            out["melangee"].append(bool(fiche.melangee[h]))
            out["diluee"].append(bool(fiche.diluee[h]))
    return out


def donnees_motif(obj, noms=None, langue=None):
    """Une ligne par partie et par bloc : la partie (étiquette, dimension) est-elle portée par le bloc ?"""
    nm = _noms(obj, noms)
    out = {k: [] for k in ("partie", "dimension", "bloc", "porte", "type")}
    for L, d in obj.motif.items():
        for k in range(len(nm)):
            out["partie"].append(_netiq(L, nm, langue))
            out["dimension"].append(int(d))
            out["bloc"].append(nm[k])
            out["porte"].append(k in L)
            out["type"].append(_type(L))
    return out


def donnees_scores(obj, prep=None, groupes=None):
    """Scores des individus sur toutes les composantes CoPRes (colonnes v1, v2, ...) et groupe."""
    if prep is not None:
        Z, g = prep.Xs @ obj.base, np.asarray(prep.g)
    else:
        Z, g = np.asarray(obj.base), (np.asarray(groupes) if groupes is not None else None)
    out = {f"v{h + 1}": Z[:, h] for h in range(Z.shape[1])}
    out["groupe"] = g if g is not None else np.array([""] * len(Z))
    return out


def donnees_loadings(obj, bloc=0, X=None, noms_variables=None):
    """Une ligne par variable : loading sur chaque composante (multibloc : loadings H_k du bloc ;
    multigroupe : coefficients des co-variations), et corrélation si X est donné (multibloc)."""
    if hasattr(obj, "loadings"):
        H = obj.loadings[bloc]
        cor = None
        if X is not None:
            x, V = X[bloc], obj.base
            cor = (x.T @ V) / (np.linalg.norm(x, axis=0)[:, None] * np.linalg.norm(V, axis=0)[None, :])
    else:
        H, cor = obj.base, None
    nv = noms_variables if noms_variables is not None else [str(v + 1) for v in range(H.shape[0])]
    out = {"variable": list(nv)}
    for h in range(H.shape[1]):
        out[f"loading_v{h + 1}"] = H[:, h]
        if cor is not None:
            out[f"correlation_v{h + 1}"] = cor[:, h]
    return out


def donnees_contributions(obj, composante=0, bloc=0, noms_variables=None):
    """Contribution de chaque variable à une composante (carré du loading normalisé) et signe."""
    d = donnees_loadings(obj, bloc, noms_variables=noms_variables)
    h = np.asarray(d[f"loading_v{composante + 1}"])
    c = h ** 2 / np.sum(h ** 2)
    o = np.argsort(-c)
    return {"variable": [d["variable"][v] for v in o], "contribution": c[o], "signe": np.sign(h[o])}


def donnees_stabilite(stab, langue=None):
    """Une ligne par partie et par critère : fréquence de retour."""
    out = {k: [] for k in ("partie", "dimension", "critere", "frequence")}
    for L, d in stab.dimensions.items():
        for cle, f in (("strict", stab.strict), ("etiquette", stab.etiquette), ("type", stab.type)):
            out["partie"].append(_netiq(L, stab.noms, langue))
            out["dimension"].append(int(d))
            out["critere"].append(cle)
            out["frequence"].append(float(f[L]))
    return out


def donnees_sensibilite(sens, facteur_reference=1.0):
    """Une ligne par (R, facteur) : motif, et s'il diffère du motif au facteur de référence."""
    out = {k: [] for k in ("R", "facteur", "motif", "different")}
    for (R, f), m in sorted(sens.items()):
        out["R"].append(R)
        out["facteur"].append(f)
        out["motif"].append(m)
        out["different"].append(m != sens.get((R, facteur_reference)))
    return out


def donnees_bilan(X, theta, R, methodes=("CPCA", "ACOM", "ACCPS"), departs=20):
    """Une ligne par méthode : bilan de la lecture directe, bilan CoPRes, borne de Ky Fan."""
    from .multibloc import copres, lecture_directe
    out = {k: [] for k in ("methode", "lecture_directe", "copres", "borne")}
    for m in methodes:
        c = copres(X, theta, R=R, methode=m, departs=departs)
        out["methode"].append(m)
        out["lecture_directe"].append(lecture_directe(X, c.base_methode, theta).bilan)
        out["copres"].append(c.bilan)
        out["borne"].append(c.borne)
    return out


# ============================================================================= dessin (expérimental)
def graph_saliences(obj, noms=None, titre=None, langue=None):
    """Saliences de chaque bloc sur chaque composante, avec le seuil du bloc (trait noir)."""
    plt = _plt()
    T = _THEME
    d = donnees_saliences(obj, noms, langue)
    S = np.asarray(obj.saliences)
    K, R = S.shape
    nm = _noms(obj, noms)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.45 * R * K + 2), 3.4))
    w = 0.8 / K
    for i in range(len(d["bloc"])):
        h, k = d["composante"][i] - 1, i % K
        x = h + (k - (K - 1) / 2) * w
        c = (T["propre"] if d["type"][i] == "propre" else T["commune"]) if d["porteur"][i] else T["non_porteur"]
        ax.bar(x, d["salience"][i], w * 0.88, color=c, edgecolor="white", linewidth=0.8)
        ax.plot([x - w * 0.45, x + w * 0.45], [d["seuil"][i]] * 2, color=T["encre"], lw=1.2)
        if R * K <= 30:
            ax.text(x, -0.02 * S.max(), nm[k], ha="center", va="top", fontsize=T["taille"] - 2, color=T["encre2"])
    ax.set_xticks(range(R))
    ax.set_xticklabels([f"v{h + 1}\n{_netiq(L, nm, langue)}" for h, L in enumerate(obj.etiquettes)],
                       fontsize=T["taille"] - 1)
    ax.tick_params(axis="x", length=0, pad=12 if R * K <= 30 else 3)
    ax.set_ylabel(_t("salience", langue))
    ax.set_ylim(0, S.max() * 1.08)
    ax.grid(axis="y", color=T["grille"], lw=0.6)
    ax.set_axisbelow(True)
    import matplotlib.patches as mp
    import matplotlib.lines as ml
    ax.legend(handles=[mp.Patch(color=T["commune"], label=_t("porteur_commune", langue)),
                       mp.Patch(color=T["propre"], label=_t("porteur_propre", langue)),
                       mp.Patch(color=T["non_porteur"], label=_t("non_porteur", langue)),
                       ml.Line2D([], [], color=T["encre"], lw=1.2, label=_t("seuil", langue))],
              fontsize=T["taille"] - 1.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    if titre:
        ax.set_title(titre, fontsize=T["taille"] + 1, loc="left", pad=22)
    fig.tight_layout()
    return fig


def graph_coherence(obj, plan=(0, 1), bloc=None, noms=None, langue=None):
    """Cohérence dans le plan de deux composantes : ellipse de chaque bloc (sa projection sur un
    axe est la racine de la salience du bloc) et cercle de rayon racine du seuil. `bloc` : indice
    (ou liste d'indices) des blocs à dessiner, tous par défaut, un panneau par bloc."""
    plt = _plt()
    T = _THEME
    dd = donnees_coherence(obj, plan, noms)
    blocs = range(len(dd)) if bloc is None else ([bloc] if np.isscalar(bloc) else list(bloc))
    dd_sel = [dd[k] for k in blocs]
    lim = 1.15 * max(max(np.abs(np.r_[e["ellipse_x"], e["ellipse_y"]]).max(), np.sqrt(e["seuil"])) for e in dd)
    fig, axes = plt.subplots(1, len(dd_sel), figsize=(max(2.6 * len(dd_sel) + 0.2, 4.2), 3.0), squeeze=False)
    for e, ax in zip(dd_sel, axes[0]):
        c = {"porte_tout": T["commune"], "porte_rien": T["non_porteur"], "porte_partie": T["melange"]}[e["etat"]]
        ax.fill(e["ellipse_x"], e["ellipse_y"], color=c, alpha=0.35, lw=0)
        ax.plot(e["ellipse_x"], e["ellipse_y"], color=c if e["etat"] != "porte_rien" else T["residuel"], lw=1.5)
        ax.plot(e["cercle_x"], e["cercle_y"], color=T["encre"], lw=1, ls="--")
        ax.axhline(0, color=T["grille"], lw=0.6, zorder=0)
        ax.axvline(0, color=T["grille"], lw=0.6, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"v{plan[0] + 1}")
        ax.set_ylabel(f"v{plan[1] + 1}")
        ax.set_title(f"{e['bloc']} : {_t(e['etat'], langue)}" if (langue or _LANGUE["valeur"]) == "fr"
                     else f"{e['bloc']}: {_t(e['etat'], langue)}", fontsize=T["taille"] - 0.5, loc="left")
        ax.tick_params(labelsize=T["taille"] - 2)
    fig.suptitle(_t("ellipse_titre", langue), fontsize=T["taille"], x=0.01, ha="left")
    fig.tight_layout()
    return fig


def graph_fiche(fiche, noms=None, etiquettes_composantes=None, langue=None):
    """Composition de chaque composante d'une méthode dans la base CoPRes : étiquette principale
    (bleu ou vert), mélange (orange), part résiduelle (gris) ; mu* et r à droite."""
    plt = _plt()
    T = _THEME
    d = donnees_fiche(fiche, noms, langue)
    H = len(fiche.composition)
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * H + 1.6))
    gauche = np.zeros(H)
    for i in range(len(d["part"])):
        h, v = d["composante"][i] - 1, d["part"][i]
        if v < 0.003:
            continue
        col = (_couleur_type(d["type"][i]) if d["role"][i] == "principale"
               else (T["residuel"] if d["role"][i] == "residuelle" else T["melange"]))
        y = H - 1 - h
        ax.barh(y, v, 0.6, left=gauche[h], color=col, edgecolor="white", linewidth=1.5)
        if v >= 0.12:
            pc = f"{100 * v:.0f} %" if (langue or _LANGUE["valeur"]) == "fr" else f"{100 * v:.0f}%"
            ax.text(gauche[h] + v / 2, y, f"{d['etiquette'][i]} {pc}", ha="center", va="center",
                    fontsize=T["taille"] - 1.5, color="white")
        gauche[h] += v
    for h in range(H):
        drap = ((" " + _t("melangee", langue)) if fiche.melangee[h] else "") + ((" " + _t("diluee", langue)) if fiche.diluee[h] else "")
        ax.text(1.03, H - 1 - h, f"μ* = {_nombre(fiche.melange[h], 2, langue)}   r = {_nombre(fiche.residuel[h], 2, langue)}{drap}",
                va="center", fontsize=T["taille"] - 1.5, color=T["encre"])
    ax.set_yticks(range(H))
    lab = etiquettes_composantes or [f"u{h + 1}" for h in range(H)]
    ax.set_yticklabels(lab[::-1])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", _nombre(0.5, 1, langue), "1"])
    ax.set_xlabel(_t("part_composante", langue))
    import matplotlib.patches as mp
    fig.legend(handles=[mp.Patch(color=T["commune"], label=_t("principale_commune", langue)),
                        mp.Patch(color=T["propre"], label=_t("principale_propre", langue)),
                        mp.Patch(color=T["melange"], label=_t("melange", langue)),
                        mp.Patch(color=T["residuel"], label=_t("residuelle", langue))],
               fontsize=T["taille"] - 2, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.06, 0.0))
    fig.subplots_adjust(left=0.08, right=0.68, top=0.95, bottom=0.62 / (0.55 * H + 1.6) + 0.12)
    return fig


def graph_motif(obj, noms=None, langue=None):
    """Schéma du motif : une ligne par partie, une colonne par bloc ; case pleine si le bloc la porte."""
    plt = _plt()
    T = _THEME
    d = donnees_motif(obj, noms, langue)
    nm = _noms(obj, noms)
    K = len(nm)
    parts = list(dict.fromkeys(d["partie"]))
    fig, ax = plt.subplots(figsize=(0.6 * K + 2.6, 0.45 * len(parts) + 0.9))
    for i in range(len(d["partie"])):
        y = len(parts) - 1 - parts.index(d["partie"][i])
        k = i % K
        porte = d["porte"][i]
        c = _couleur_type(d["type"][i]) if porte else "white"
        ax.add_patch(plt.Rectangle((k + 0.08, y + 0.1), 0.84, 0.8, facecolor=c,
                                   edgecolor=c if porte else T["non_porteur"], lw=1))
        if k == 0:
            ax.text(K + 0.2, y + 0.5, f"{_t('dim', langue)} {d['dimension'][i]}", va="center",
                    fontsize=T["taille"] - 1, color=T["encre2"])
    ax.set_xlim(0, K + 1.2)
    ax.set_ylim(0, len(parts))
    ax.set_xticks(np.arange(K) + 0.5)
    ax.set_xticklabels(nm)
    ax.set_yticks(np.arange(len(parts)) + 0.5)
    ax.set_yticklabels(parts[::-1])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    fig.tight_layout()
    return fig


def _ellipse(ax, Z, couleur):
    if len(Z) < 3:
        return
    m = Z.mean(0)
    w, v = np.linalg.eigh(np.cov(Z.T))
    t = np.linspace(0, 2 * np.pi, 200)
    E = (v @ np.diag(np.sqrt(np.clip(w, 0, None) * 5.991)) @ np.vstack([np.cos(t), np.sin(t)])).T + m
    ax.plot(E[:, 0], E[:, 1], color=couleur, lw=1.2)


def graph_scores(obj, composantes=(0, 1), groupes=None, prep=None, ellipses=True, etiquettes_points=None,
                 noms=None, langue=None):
    """Carte des individus sur deux composantes CoPRes, colorés par groupe (multigroupe : `prep` ;
    multibloc : `groupes`, facultatif), avec ellipses à 95 % (loi normale)."""
    plt = _plt()
    T = _THEME
    d = donnees_scores(obj, prep, groupes)
    i, j = composantes
    xi, yj = np.asarray(d[f"v{i + 1}"]), np.asarray(d[f"v{j + 1}"])
    g = np.asarray(d["groupe"])
    nm = _noms(obj, noms)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    niveaux = list(dict.fromkeys(g.tolist()))
    for c, k in enumerate(niveaux):
        m = g == k
        col = T["groupes"][c % len(T["groupes"])] if len(niveaux) > 1 else T["commune"]
        grand = len(g) > 200
        ax.scatter(xi[m], yj[m], s=14 if grand else 24, color=col, alpha=0.7 if grand else 1, edgecolor="white",
                   linewidth=0.5, label=str(k) if len(niveaux) > 1 else None)
        if ellipses and len(niveaux) > 1:
            _ellipse(ax, np.column_stack([xi[m], yj[m]]), col)
    if len(niveaux) > 1:
        ax.legend(fontsize=T["taille"] - 1.5, frameon=False, markerscale=1.3)
    if etiquettes_points is not None:
        for p, t in enumerate(etiquettes_points):
            ax.annotate(str(t), (xi[p], yj[p]), fontsize=T["taille"] - 2.5, color=T["encre2"], xytext=(3, 2),
                        textcoords="offset points")
    et = obj.etiquettes
    ax.set_xlabel(f"v{i + 1} {_netiq(et[i], nm, langue)}")
    ax.set_ylabel(f"v{j + 1} {_netiq(et[j], nm, langue)}")
    ax.axhline(0, color=T["grille"], lw=0.6, zorder=0)
    ax.axvline(0, color=T["grille"], lw=0.6, zorder=0)
    fig.tight_layout()
    return fig


def graph_scores_parties(obj, groupes=None, prep=None, partie=None, noms=None, langue=None):
    """Une carte par partie (hors résiduelle) : plan des deux premières composantes si la partie
    est de dimension au moins 2, sinon scores par groupe (points décalés). `partie` : étiquette
    (tuple) d'une seule partie à dessiner."""
    plt = _plt()
    T = _THEME
    d = donnees_scores(obj, prep, groupes)
    Z = np.column_stack([d[f"v{h + 1}"] for h in range(len(obj.etiquettes))])
    g = np.asarray(d["groupe"])
    nm = _noms(obj, noms)
    parts = [(L, idx) for L, idx in _parties(obj).items() if L and (partie is None or L == tuple(partie))]
    fig, axes = plt.subplots(1, len(parts), figsize=(3.0 * len(parts) + 0.2, 3.0), squeeze=False)
    niv = list(dict.fromkeys(g.tolist()))
    rng = np.random.default_rng(0)
    for (L, idx), ax in zip(parts, axes[0]):
        for c, k in enumerate(niv):
            m = g == k
            col = T["groupes"][c % len(T["groupes"])] if len(niv) > 1 else T["commune"]
            if len(idx) >= 2:
                ax.scatter(Z[m, idx[0]], Z[m, idx[1]], s=8, color=col, alpha=0.6, linewidth=0)
            else:
                ax.scatter(Z[m, idx[0]], c + rng.uniform(-0.25, 0.25, m.sum()), s=8, color=col, alpha=0.6, linewidth=0)
        if len(idx) >= 2:
            ax.set_xlabel(f"v{idx[0] + 1}")
            ax.set_ylabel(f"v{idx[1] + 1}")
        else:
            ax.set_yticks(range(len(niv)))
            ax.set_yticklabels([str(k) for k in niv])
            ax.set_xlabel(f"v{idx[0] + 1}")
        ax.set_title(f"{_netiq(L, nm, langue)} ({_t('dim', langue)} {len(idx)})", fontsize=T["taille"], loc="left")
        ax.axvline(0, color=T["grille"], lw=0.6, zorder=0)
    fig.tight_layout()
    return fig


def graph_loadings(obj, bloc=0, composantes=(0, 1), type="cercle", X=None, noms_variables=None, axe=None,
                   nb_noms=20, noms=None, langue=None):
    """Variables d'un bloc sur les composantes CoPRes : type "cercle" (cercle des corrélations si X
    est donné en multibloc, coefficients sinon) ou "spectre" (loadings en fonction de `axe`)."""
    plt = _plt()
    T = _THEME
    d = donnees_loadings(obj, bloc, X, noms_variables)
    nm = _noms(obj, noms)
    i, j = composantes[0], composantes[1] if len(composantes) > 1 else composantes[0]
    if type == "spectre":
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        a = np.arange(len(d["variable"])) if axe is None else np.asarray(axe)
        for c, h in enumerate(composantes):
            ax.plot(a, d[f"loading_v{h + 1}"], lw=1.4, color=T["groupes"][c % len(T["groupes"])],
                    label=f"v{h + 1} {_netiq(obj.etiquettes[h], nm, langue)}")
        ax.axhline(0, color=T["grille"], lw=0.6, zorder=0)
        ax.set_ylabel(_t("loading", langue))
        ax.legend(fontsize=T["taille"] - 1.5, frameon=False)
        fig.tight_layout()
        return fig
    cle = "correlation" if f"correlation_v{i + 1}" in d else "loading"
    mx, my = np.asarray(d[f"{cle}_v{i + 1}"]), np.asarray(d[f"{cle}_v{j + 1}"])
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    if cle == "correlation":
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color=T["grille"], lw=0.8)
        b = 1.08
    else:
        b = 1.25 * max(np.abs(np.r_[mx, my]).max(), 1e-12)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.axhline(0, color=T["grille"], lw=0.6, zorder=0)
    ax.axvline(0, color=T["grille"], lw=0.6, zorder=0)
    for v in range(len(mx)):
        ax.annotate("", xy=(mx[v], my[v]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=T["commune"], lw=0.8, mutation_scale=7))
    for v in np.argsort(-np.hypot(mx, my))[:nb_noms]:
        r = max(np.hypot(mx[v], my[v]), 1e-12)
        ax.annotate(d["variable"][v], (mx[v], my[v]), xytext=(6 * mx[v] / r, 6 * my[v] / r), textcoords="offset points",
                    fontsize=T["taille"] - 2, color=T["encre"], ha="left" if mx[v] >= 0 else "right", va="center")
    ax.set_xlabel(f"v{i + 1} {_netiq(obj.etiquettes[i], nm, langue)}")
    ax.set_ylabel(f"v{j + 1} {_netiq(obj.etiquettes[j], nm, langue)}")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def graph_contributions(obj, composante=0, bloc=0, nb=15, noms_variables=None, noms=None, langue=None):
    """Variables les plus contributives à une composante (carré du loading), avec le signe."""
    plt = _plt()
    T = _THEME
    d = donnees_contributions(obj, composante, bloc, noms_variables)
    n = min(nb, len(d["variable"]))
    c, sg, nv = d["contribution"][:n][::-1], d["signe"][:n][::-1], d["variable"][:n][::-1]
    fig, ax = plt.subplots(figsize=(4.8, 0.26 * n + 1.0))
    ax.barh(range(n), c, 0.7, color=[T["commune"] if s >= 0 else T["melange"] for s in sg])
    ax.set_yticks(range(n))
    ax.set_yticklabels(nv, fontsize=T["taille"] - 1.5)
    ax.set_xlabel(_t("contribution", langue))
    ax.grid(axis="x", color=T["grille"], lw=0.6)
    ax.set_axisbelow(True)
    nm = _noms(obj, noms)
    ax.set_title(f"v{composante + 1} {_netiq(obj.etiquettes[composante], nm, langue)} ; {_t('signe', langue)}",
                 fontsize=T["taille"] - 0.5, loc="left")
    fig.tight_layout()
    return fig


def graph_stabilite(stab, langue=None):
    """Fréquences bootstrap de chaque partie selon les trois critères."""
    plt = _plt()
    T = _THEME
    d = donnees_stabilite(stab, langue)
    parts = list(dict.fromkeys(d["partie"]))
    n = len(parts)
    fig, ax = plt.subplots(figsize=(6.0, 0.55 * n + 1.3))
    for c, cle in enumerate(("strict", "etiquette", "type")):
        idx = [i for i in range(len(d["partie"])) if d["critere"][i] == cle]
        y = np.array([n - 1 - parts.index(d["partie"][i]) for i in idx]) + (1 - c) * 0.26
        ax.barh(y, [100 * d["frequence"][i] for i in idx], 0.24, color=T["seq"][c], label=_t(cle, langue),
                edgecolor="white", linewidth=0.8)
    dims = {p: d["dimension"][d["partie"].index(p)] for p in parts}
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([f"{p} ({dims[p]})" for p in parts])
    ax.set_xlim(0, 100)
    ax.set_xlabel(_t("stab_axe", langue))
    ax.grid(axis="x", color=T["grille"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=T["taille"] - 1.5, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22 if n > 3 else -0.35), title=_t("critere", langue), title_fontsize=T["taille"] - 1.5)
    ax.set_title(_t("stab_titre", langue, nb=stab.nb, p=f"{100 * stab.motif_retrouve:.0f}"),
                 fontsize=T["taille"] - 0.5, loc="left")
    fig.tight_layout()
    return fig


def graph_sensibilite(sens, facteur_reference=1.0, langue=None):
    """Grille des motifs selon R et le facteur des seuils ; case bleue si le motif diffère de
    celui du facteur de référence pour le même R."""
    plt = _plt()
    T = _THEME
    d = donnees_sensibilite(sens, facteur_reference)
    Rs, fs = sorted(set(d["R"])), sorted(set(d["facteur"]))
    fig, ax = plt.subplots(figsize=(2.4 * len(fs) + 0.8, 0.85 * len(Rs) + 0.8))
    for R, f, m, diff in zip(d["R"], d["facteur"], d["motif"], d["different"]):
        a, b = Rs.index(R), fs.index(f)
        y = len(Rs) - 1 - a
        ax.add_patch(plt.Rectangle((b, y), 0.97, 0.94, facecolor="#cde2fb" if diff else "#f2f1ee", lw=0))
        if (langue or _LANGUE["valeur"]) == "en":
            m = m.replace("résiduelle", "residual")
        ax.text(b + 0.48, y + 0.47, m.replace("), (", ")\n("), ha="center", va="center",
                fontsize=T["taille"] - 2.7, color=T["encre"])
    ax.set_xlim(0, len(fs))
    ax.set_ylim(0, len(Rs))
    ax.set_xticks(np.arange(len(fs)) + 0.48)
    ax.set_xticklabels([_t("seuil_facteur", langue, f=_nombre(f, 2, langue).rstrip("0").rstrip(",.")) for f in fs])
    ax.set_yticks(np.arange(len(Rs)) + 0.47)
    ax.set_yticklabels([f"R = {R}" for R in Rs][::-1])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_title(_t("sens_titre", langue), fontsize=T["taille"] - 1, loc="left", pad=22)
    fig.tight_layout()
    return fig


def graph_bilan(X=None, theta=None, R=None, methodes=("CPCA", "ACOM", "ACCPS"), departs=20, donnees=None, langue=None):
    """Bilan Phi de la lecture directe et de la base CoPRes de chaque méthode, avec la borne de Ky
    Fan de son sous-espace. Donner (X, theta, R) ou un tableau `donnees` issu de donnees_bilan."""
    plt = _plt()
    T = _THEME
    d = donnees if donnees is not None else donnees_bilan(X, theta, R, methodes, departs)
    fig, ax = plt.subplots(figsize=(1.6 * len(d["methode"]) + 1.6, 3.2))
    for i, m in enumerate(d["methode"]):
        dl, cb, bo = d["lecture_directe"][i], d["copres"][i], d["borne"][i]
        ax.bar(i - 0.18, dl, 0.34, color=T["non_porteur"], label=_t("lecture_directe", langue) if i == 0 else None)
        ax.bar(i + 0.18, cb, 0.34, color=T["commune"], label=_t("copres", langue) if i == 0 else None)
        ax.plot([i - 0.38, i + 0.38], [bo, bo], color=T["encre"], lw=1.3, label=_t("borne", langue) if i == 0 else None)
        ax.text(i + 0.18, cb * 0.97, _nombre(cb, 3, langue), ha="center", va="top", fontsize=T["taille"] - 2, color="white")
        ax.text(i - 0.18, dl * 0.97, _nombre(dl, 3, langue), ha="center", va="top", fontsize=T["taille"] - 2, color=T["encre"])
    ax.set_xticks(range(len(d["methode"])))
    ax.set_xticklabels(d["methode"])
    ax.set_ylabel(_t("bilan", langue))
    ax.legend(fontsize=T["taille"] - 1.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis="y", color=T["grille"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
