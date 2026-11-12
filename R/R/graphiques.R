# Graphiques de CoPRes (ggplot2, paquet suggéré). Trois couches :
#  - calcul (copres, fiche, stabilite...) : stable ;
#  - données des graphiques (donnees_*) : data.frame prêt à tracer, stable ;
#  - dessin (graph_*) : proposition de mise en forme, EXPÉRIMENTALE ; chaque fonction rend un
#    objet ggplot indépendant (graph_coherence et graph_scores_parties : une liste si plusieurs
#    panneaux, à assembler par exemple avec patchwork::wrap_plots).
# Langue : options(CoPRes.langue = "en") ou langue("en"), ou l'argument langue = de chaque fonction ;
# textes dans inst/extdata/textes.csv, fichier commun avec la version Python.
# Thème : theme_copres(commune = ..., taille = ...).

# ------------------------------------------------------------------ langue et thème
.THEME_DEFAUT <- list(commune = "#2a78d6", propre = "#199e70", non_porteur = "#d9d8d4", melange = "#eb6834",
                      residuel = "#8a8984", encre = "#0b0b0b", encre2 = "#52514e", grille = "#e6e5e1",
                      groupes = c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
                      seq = c("#9ec5f4", "#5598e7", "#1c5cab"), taille = 9)
.etat <- new.env()

langue <- function(valeur = NULL) {
  if (!is.null(valeur)) {
    if (!valeur %in% c("fr", "en")) stop("langue : 'fr' ou 'en'")
    options(CoPRes.langue = valeur)
  }
  getOption("CoPRes.langue", "fr")
}

theme_copres <- function(..., defaut = FALSE) {
  if (is.null(.etat$theme) || defaut) .etat$theme <- .THEME_DEFAUT
  r <- list(...)
  for (k in names(r)) {
    if (!k %in% names(.THEME_DEFAUT)) stop("clé de thème inconnue : ", k)
    .etat$theme[[k]] <- r[[k]]
  }
  invisible(.etat$theme)
}

.T <- function() { if (is.null(.etat$theme)) .etat$theme <- .THEME_DEFAUT; .etat$theme }

.t <- function(cle, lg = NULL, ...) {
  if (is.null(.etat$textes)) {
    f <- system.file("extdata", "textes.csv", package = "CoPRes")
    .etat$textes <- utils::read.csv(f, sep = ";", fileEncoding = "UTF-8", stringsAsFactors = FALSE)
  }
  lg <- if (is.null(lg)) langue() else lg
  s <- .etat$textes[.etat$textes$cle == cle, lg]
  a <- list(...)
  for (k in names(a)) s <- gsub(paste0("{", k, "}"), a[[k]], s, fixed = TRUE)
  s
}

.nombre <- function(x, n, lg = NULL) {
  s <- formatC(x, format = "f", digits = n)
  if ((if (is.null(lg)) langue() else lg) == "fr") gsub(".", ",", s, fixed = TRUE) else s
}

.netiq <- function(L, noms, lg) {
  if (!length(L)) return(.t("residuelle", lg))
  paste0("{", paste(if (is.null(noms)) L else noms[L], collapse = ","), "}")
}

.noms <- function(obj, noms = NULL) {
  if (!is.null(noms)) return(noms)
  if (!is.null(obj$groupes)) return(obj$groupes)
  as.character(seq_len(nrow(obj$saliences)))
}

.type <- function(L) if (!length(L)) "residuelle" else if (length(L) == 1) "propre" else "commune"
.cle_vers_L <- function(k) if (is.na(k) || k == "") integer(0) else as.integer(strsplit(k, ",")[[1]])

.gg <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) stop("installer le paquet ggplot2 pour les graphiques")
}

.theme_gg <- function() {
  T <- .T()
  ggplot2::theme_minimal(base_size = T$taille + 2) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), panel.grid.major.x = ggplot2::element_blank(),
                   panel.grid.major.y = ggplot2::element_line(colour = T$grille, linewidth = 0.3),
                   axis.line = ggplot2::element_line(colour = T$encre2, linewidth = 0.3),
                   legend.position = "top", legend.title = ggplot2::element_blank(),
                   text = ggplot2::element_text(colour = T$encre))
}

# ================================================================== données des graphiques
donnees_saliences <- function(obj, noms = NULL, langue = NULL) {
  S <- obj$saliences; nm <- .noms(obj, noms); K <- nrow(S)
  do.call(rbind, lapply(seq_along(obj$etiquettes), function(h) {
    L <- obj$etiquettes[[h]]
    data.frame(composante = h, etiquette = .netiq(L, nm, langue), bloc = nm, salience = S[, h],
               seuil = obj$seuils, porteur = seq_len(K) %in% L, type = .type(L), stringsAsFactors = FALSE)
  }))
}

donnees_coherence <- function(obj, plan = c(1, 2), noms = NULL, nb_points = 361) {
  res <- obj$lode; nm <- .noms(obj, noms)
  Q <- res$base[, plan, drop = FALSE]
  t <- seq(0, 2 * pi, length.out = nb_points); C <- rbind(cos(t), sin(t))
  do.call(rbind, lapply(seq_along(res$matrices), function(k) {
    e <- eigen(t(Q) %*% res$matrices[[k]] %*% Q, symmetric = TRUE)
    w <- rev(e$values)
    E <- e$vectors %*% diag(sqrt(pmax(e$values, 0))) %*% t(e$vectors) %*% C
    th <- obj$seuils[k]
    etat <- if (w[1] >= th) "porte_tout" else if (w[2] <= th) "porte_rien" else "porte_partie"
    rbind(data.frame(bloc = nm[k], trace = "ellipse", x = E[1, ], y = E[2, ], lambda_min = w[1], lambda_max = w[2],
                     seuil = th, etat = etat, stringsAsFactors = FALSE),
          data.frame(bloc = nm[k], trace = "cercle", x = sqrt(th) * C[1, ], y = sqrt(th) * C[2, ], lambda_min = w[1],
                     lambda_max = w[2], seuil = th, etat = etat, stringsAsFactors = FALSE))
  }))
}

donnees_fiche <- function(fiche, noms = NULL, langue = NULL) {
  do.call(rbind, lapply(seq_along(fiche$composition), function(h) {
    c <- fiche$composition[[h]]; kp <- fiche$principale[h]
    cles <- c(if (!is.na(kp)) kp, setdiff(names(c)[names(c) != ""], kp), if ("" %in% names(c)) "")
    data.frame(composante = h, etiquette = vapply(cles, function(k) .netiq(.cle_vers_L(k), noms, langue), ""),
               part = as.numeric(c[cles]),
               role = ifelse(cles == kp & !is.na(kp), "principale", ifelse(cles == "", "residuelle", "melange")),
               type = vapply(cles, function(k) .type(.cle_vers_L(k)), ""),
               mu = fiche$melange[h], r = fiche$residuel[h], melangee = fiche$melangee[h], diluee = fiche$diluee[h],
               row.names = NULL, stringsAsFactors = FALSE)
  }))
}

donnees_motif <- function(obj, noms = NULL, langue = NULL) {
  nm <- .noms(obj, noms); m <- obj$motif
  do.call(rbind, lapply(seq_len(nrow(m)), function(i) {
    L <- m$etiquette[[i]]
    data.frame(partie = .netiq(L, nm, langue), dimension = m$multiplicite[i], bloc = nm,
               porte = seq_along(nm) %in% L, type = .type(L), stringsAsFactors = FALSE)
  }))
}

donnees_scores <- function(obj, prep = NULL, groupes = NULL) {
  if (!is.null(prep)) { Z <- prep$Xs %*% obj$base; g <- prep$g } else { Z <- obj$base; g <- groupes }
  d <- as.data.frame(Z); names(d) <- paste0("v", seq_len(ncol(Z)))
  d$groupe <- if (is.null(g)) "" else as.character(g)
  d
}

donnees_loadings <- function(obj, bloc = 1, X = NULL, noms_variables = NULL) {
  if (!is.null(obj$loadings)) {
    H <- obj$loadings[[bloc]]; cor <- NULL
    if (!is.null(X)) {
      x <- X[[bloc]]; V <- obj$base
      cor <- crossprod(x, V) / outer(sqrt(colSums(x^2)), sqrt(colSums(V^2)))
    }
  } else { H <- obj$base; cor <- NULL }
  d <- data.frame(variable = if (is.null(noms_variables)) as.character(seq_len(nrow(H))) else noms_variables,
                  stringsAsFactors = FALSE)
  for (h in seq_len(ncol(H))) {
    d[[paste0("loading_v", h)]] <- H[, h]
    if (!is.null(cor)) d[[paste0("correlation_v", h)]] <- cor[, h]
  }
  d
}

donnees_contributions <- function(obj, composante = 1, bloc = 1, noms_variables = NULL) {
  d <- donnees_loadings(obj, bloc, noms_variables = noms_variables)
  h <- d[[paste0("loading_v", composante)]]; c <- h^2 / sum(h^2); o <- order(-c)
  data.frame(variable = d$variable[o], contribution = c[o], signe = sign(h[o]), stringsAsFactors = FALSE)
}

donnees_stabilite <- function(stab, langue = NULL) {
  nm <- vapply(stab$etiquettes, .netiq, "", noms = stab$noms, lg = langue)
  do.call(rbind, lapply(c("strict", "etiquette", "type"), function(cr)
    data.frame(partie = nm, dimension = as.integer(stab$dimensions), critere = cr,
               frequence = as.numeric(stab[[cr]]), stringsAsFactors = FALSE)))
}

donnees_sensibilite <- function(sens, facteur_reference = 1) {
  d <- sens[order(sens$R, sens$facteur), ]
  rownames(d) <- NULL
  ref <- vapply(seq_len(nrow(d)), function(i) {
    m <- d$motif[d$R == d$R[i] & abs(d$facteur - facteur_reference) < 1e-12]
    if (length(m)) m[1] else NA_character_
  }, "")
  d$different <- is.na(ref) | d$motif != ref
  d
}

donnees_bilan <- function(X, theta, R, methodes = c("CPCA", "ACOM", "ACCPS"), departs = 20) {
  do.call(rbind, lapply(methodes, function(m) {
    c <- copres(X, theta, R = R, methode = m, departs = departs)
    data.frame(methode = m, lecture_directe = lecture_directe(X, c$base_methode, theta)$bilan,
               copres = c$bilan, borne = c$borne, stringsAsFactors = FALSE)
  }))
}

# ================================================================== dessin (expérimental)
graph_saliences <- function(obj, noms = NULL, titre = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_saliences(obj, noms, langue); nm <- .noms(obj, noms); K <- length(nm); w <- 0.8 / K
  d$x <- d$composante + ((match(d$bloc, nm) - 1) - (K - 1) / 2) * w
  d$categorie <- ifelse(!d$porteur, "non_porteur", ifelse(d$type == "propre", "propre", "commune"))
  lab <- c(commune = .t("porteur_commune", langue), propre = .t("porteur_propre", langue), non_porteur = .t("non_porteur", langue))
  axe <- vapply(seq_along(obj$etiquettes), function(h) paste0("v", h, "\n", .netiq(obj$etiquettes[[h]], nm, langue)), "")
  ggplot2::ggplot(d) +
    ggplot2::geom_col(ggplot2::aes(x = x, y = salience, fill = categorie), width = w * 0.88) +
    ggplot2::geom_segment(ggplot2::aes(x = x - w * 0.45, xend = x + w * 0.45, y = seuil, yend = seuil, linetype = "seuil"),
                          colour = T$encre, linewidth = 0.5) +
    ggplot2::scale_linetype_manual(values = c(seuil = "solid"), labels = .t("seuil", langue)) +
    ggplot2::geom_text(ggplot2::aes(x = x, y = -0.03 * max(salience), label = bloc), size = (T$taille - 2) / 2.8, colour = T$encre2, vjust = 1) +
    ggplot2::scale_fill_manual(values = c(commune = T$commune, propre = T$propre, non_porteur = T$non_porteur), labels = lab,
                               breaks = names(lab)) +
    ggplot2::scale_x_continuous(breaks = seq_along(axe), labels = axe) +
    ggplot2::labs(x = NULL, y = .t("salience", langue), title = titre) +
    .theme_gg()
}

graph_coherence <- function(obj, plan = c(1, 2), bloc = NULL, noms = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_coherence(obj, plan, noms); nm <- .noms(obj, noms)
  lim <- 1.15 * max(abs(c(d$x, d$y)))
  sep <- if ((if (is.null(langue)) CoPRes::langue() else langue) == "fr") " : " else ": "
  un <- function(b) {
    e <- d[d$bloc == b, ]; et <- e$etat[1]
    col <- c(porte_tout = T$commune, porte_rien = T$residuel, porte_partie = T$melange)[[et]]
    fil <- c(porte_tout = T$commune, porte_rien = T$non_porteur, porte_partie = T$melange)[[et]]
    ggplot2::ggplot() +
      ggplot2::geom_hline(yintercept = 0, colour = T$grille) + ggplot2::geom_vline(xintercept = 0, colour = T$grille) +
      ggplot2::geom_polygon(data = e[e$trace == "ellipse", ], ggplot2::aes(x, y), fill = fil, alpha = 0.35, colour = col) +
      ggplot2::geom_path(data = e[e$trace == "cercle", ], ggplot2::aes(x, y), linetype = "dashed", colour = T$encre) +
      ggplot2::coord_equal(xlim = c(-lim, lim), ylim = c(-lim, lim)) +
      ggplot2::labs(x = paste0("v", plan[1]), y = paste0("v", plan[2]), title = paste0(b, sep, .t(et, langue)),
                    caption = .t("ellipse_titre", langue)) +
      .theme_gg()
  }
  sel <- if (is.null(bloc)) nm else nm[bloc]
  if (length(sel) == 1) un(sel) else stats::setNames(lapply(sel, un), sel)
}

graph_fiche <- function(fiche, noms = NULL, etiquettes_composantes = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_fiche(fiche, noms, langue); d <- d[d$part >= 0.003, ]
  H <- length(fiche$composition)
  d$fin <- ave(d$part, d$composante, FUN = cumsum); d$debut <- d$fin - d$part
  d$y <- H + 1 - d$composante
  d$couleur <- ifelse(d$role == "principale", ifelse(d$type == "propre", "propre", "commune"), d$role)
  pc <- if ((if (is.null(langue)) CoPRes::langue() else langue) == "fr") " %" else "%"
  d$texte <- ifelse(d$part >= 0.12, paste0(d$etiquette, " ", round(100 * d$part), pc), "")
  droite <- data.frame(y = H + 1 - seq_len(H), texte = paste0(
    "μ* = ", .nombre(fiche$melange, 2, langue), "   r = ", .nombre(fiche$residuel, 2, langue),
    ifelse(fiche$melangee, paste0(" ", .t("melangee", langue)), ""), ifelse(fiche$diluee, paste0(" ", .t("diluee", langue)), "")))
  lab <- if (is.null(etiquettes_composantes)) paste0("u", seq_len(H)) else etiquettes_composantes
  vals <- c(commune = T$commune, propre = T$propre, melange = T$melange, residuelle = T$residuel)
  noms_l <- c(commune = .t("principale_commune", langue), propre = .t("principale_propre", langue),
              melange = .t("melange", langue), residuelle = .t("residuelle", langue))
  ggplot2::ggplot(d) +
    ggplot2::geom_rect(ggplot2::aes(xmin = debut, xmax = fin, ymin = y - 0.3, ymax = y + 0.3, fill = couleur), colour = "white") +
    ggplot2::geom_text(ggplot2::aes(x = (debut + fin) / 2, y = y, label = texte), colour = "white", size = (T$taille - 1.5) / 2.8) +
    ggplot2::geom_text(data = droite, ggplot2::aes(x = 1.03, y = y, label = texte), hjust = 0, size = (T$taille - 1.5) / 2.8) +
    ggplot2::scale_fill_manual(values = vals, labels = noms_l, breaks = names(vals)) +
    ggplot2::scale_y_continuous(breaks = H + 1 - seq_len(H), labels = lab) +
    ggplot2::scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0", .nombre(0.5, 1, langue), "1")) +
    ggplot2::coord_cartesian(xlim = c(0, 1), clip = "off") +
    ggplot2::labs(x = .t("part_composante", langue), y = NULL) +
    .theme_gg() + ggplot2::theme(legend.position = "bottom", plot.margin = ggplot2::margin(5, 150, 5, 5))
}

graph_motif <- function(obj, noms = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_motif(obj, noms, langue); nm <- .noms(obj, noms)
  parts <- unique(d$partie)
  d$x <- match(d$bloc, nm); d$y <- length(parts) + 1 - match(d$partie, parts)
  d$fond <- ifelse(d$porte, ifelse(d$type == "propre", T$propre, T$commune), "white")
  dims <- d[!duplicated(d$partie), ]
  ggplot2::ggplot(d) +
    ggplot2::geom_tile(ggplot2::aes(x = x, y = y), fill = d$fond, colour = ifelse(d$porte, d$fond, T$non_porteur),
                       width = 0.84, height = 0.8) +
    ggplot2::geom_text(data = dims, ggplot2::aes(x = length(nm) + 0.8, y = y, label = paste(.t("dim", langue), dimension)),
                       hjust = 0, colour = T$encre2, size = (T$taille - 1) / 2.8) +
    ggplot2::scale_x_continuous(breaks = seq_along(nm), labels = nm, position = "top", limits = c(0.5, length(nm) + 1.8)) +
    ggplot2::scale_y_continuous(breaks = rev(seq_along(parts)), labels = parts) +
    ggplot2::labs(x = NULL, y = NULL) + ggplot2::theme_void(base_size = T$taille + 2) +
    ggplot2::theme(axis.text = ggplot2::element_text(colour = T$encre))
}

graph_scores <- function(obj, composantes = c(1, 2), groupes = NULL, prep = NULL, ellipses = TRUE,
                         etiquettes_points = NULL, noms = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_scores(obj, prep, groupes); nm <- .noms(obj, noms)
  d$x <- d[[paste0("v", composantes[1])]]; d$y <- d[[paste0("v", composantes[2])]]
  niv <- unique(d$groupe); d$groupe <- factor(d$groupe, levels = niv)
  p <- ggplot2::ggplot(d, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_hline(yintercept = 0, colour = T$grille) + ggplot2::geom_vline(xintercept = 0, colour = T$grille)
  if (length(niv) > 1) {
    p <- p + ggplot2::geom_point(ggplot2::aes(colour = groupe), size = if (nrow(d) > 200) 0.8 else 1.8,
                                 alpha = if (nrow(d) > 200) 0.7 else 1) +
      ggplot2::scale_colour_manual(values = rep(T$groupes, length.out = length(niv)))
    if (ellipses) p <- p + ggplot2::stat_ellipse(ggplot2::aes(colour = groupe), type = "norm", level = 0.95)
  } else p <- p + ggplot2::geom_point(colour = T$commune, size = 1.8)
  if (!is.null(etiquettes_points)) p <- p + ggplot2::geom_text(ggplot2::aes(label = etiquettes_points), size = 2.3,
                                                               hjust = -0.2, colour = T$encre2)
  et <- obj$etiquettes
  p + ggplot2::labs(x = paste("v", composantes[1], " ", .netiq(et[[composantes[1]]], nm, langue), sep = ""),
                    y = paste("v", composantes[2], " ", .netiq(et[[composantes[2]]], nm, langue), sep = "")) +
    .theme_gg() + ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())
}

graph_scores_parties <- function(obj, groupes = NULL, prep = NULL, partie = NULL, noms = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_scores(obj, prep, groupes); nm <- .noms(obj, noms)
  g <- .groupes(obj$etiquettes); Ls <- attr(g, "etiq")
  garde <- vapply(Ls, function(L) length(L) > 0 && (is.null(partie) || identical(as.integer(L), as.integer(partie))), TRUE)
  niv <- unique(d$groupe)
  un <- function(L, idx) {
    e <- d; e$groupe <- factor(e$groupe, levels = niv)
    titre <- paste0(.netiq(L, nm, langue), " (", .t("dim", langue), " ", length(idx), ")")
    cols <- ggplot2::scale_colour_manual(values = rep(T$groupes, length.out = length(niv)), guide = "none")
    if (length(idx) >= 2) {
      e$x <- e[[paste0("v", idx[1])]]; e$y <- e[[paste0("v", idx[2])]]
      ggplot2::ggplot(e, ggplot2::aes(x, y, colour = groupe)) + ggplot2::geom_point(size = 0.6, alpha = 0.6) + cols +
        ggplot2::labs(x = paste0("v", idx[1]), y = paste0("v", idx[2]), title = titre) + .theme_gg()
    } else {
      e$x <- e[[paste0("v", idx[1])]]
      ggplot2::ggplot(e, ggplot2::aes(x, groupe, colour = groupe)) +
        ggplot2::geom_point(position = ggplot2::position_jitter(width = 0, height = 0.25, seed = 1), size = 0.6, alpha = 0.6) + cols +
        ggplot2::labs(x = paste0("v", idx[1]), y = NULL, title = titre) + .theme_gg()
    }
  }
  out <- lapply(which(garde), function(i) un(Ls[[i]], g[[i]]))
  names(out) <- vapply(Ls[garde], .netiq, "", noms = nm, lg = langue)
  if (length(out) == 1) out[[1]] else out
}

graph_loadings <- function(obj, bloc = 1, composantes = c(1, 2), type = c("cercle", "spectre"), X = NULL,
                           noms_variables = NULL, axe = NULL, nb_noms = 20, noms = NULL, langue = NULL) {
  .gg(); T <- .T(); type <- match.arg(type)
  d <- donnees_loadings(obj, bloc, X, noms_variables); nm <- .noms(obj, noms)
  if (type == "spectre") {
    a <- if (is.null(axe)) seq_len(nrow(d)) else axe
    e <- do.call(rbind, lapply(composantes, function(h) data.frame(axe = a, loading = d[[paste0("loading_v", h)]],
                       composante = paste0("v", h, " ", .netiq(obj$etiquettes[[h]], nm, langue)))))
    e$composante <- factor(e$composante, levels = unique(e$composante))
    return(ggplot2::ggplot(e, ggplot2::aes(axe, loading, colour = composante)) +
             ggplot2::geom_hline(yintercept = 0, colour = T$grille) + ggplot2::geom_line(linewidth = 0.6) +
             ggplot2::scale_colour_manual(values = T$groupes) + ggplot2::labs(x = NULL, y = .t("loading", langue)) + .theme_gg())
  }
  i <- composantes[1]; j <- composantes[2]
  cle <- if (paste0("correlation_v", i) %in% names(d)) "correlation" else "loading"
  e <- data.frame(variable = d$variable, x = d[[paste0(cle, "_v", i)]], y = d[[paste0(cle, "_v", j)]])
  e$montrer <- rank(-sqrt(e$x^2 + e$y^2)) <= nb_noms
  b <- if (cle == "correlation") 1.08 else 1.25 * max(abs(c(e$x, e$y)))
  p <- ggplot2::ggplot(e) + ggplot2::geom_hline(yintercept = 0, colour = T$grille) + ggplot2::geom_vline(xintercept = 0, colour = T$grille)
  if (cle == "correlation") {
    t <- seq(0, 2 * pi, length.out = 200)
    p <- p + ggplot2::geom_path(data = data.frame(x = cos(t), y = sin(t)), ggplot2::aes(x, y), colour = T$grille)
  }
  p + ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = x, yend = y), colour = T$commune, linewidth = 0.4,
                            arrow = ggplot2::arrow(length = ggplot2::unit(0.12, "cm"), type = "closed")) +
    ggplot2::geom_text(data = e[e$montrer, ], ggplot2::aes(x = x * 1.08, y = y * 1.08, label = variable,
                                                          hjust = ifelse(x >= 0, 0, 1)), size = (T$taille - 2) / 2.8) +
    ggplot2::coord_equal(xlim = c(-b, b), ylim = c(-b, b)) +
    ggplot2::labs(x = paste0("v", i, " ", .netiq(obj$etiquettes[[i]], nm, langue)),
                  y = paste0("v", j, " ", .netiq(obj$etiquettes[[j]], nm, langue))) +
    .theme_gg() + ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())
}

graph_contributions <- function(obj, composante = 1, bloc = 1, nb = 15, noms_variables = NULL, noms = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- utils::head(donnees_contributions(obj, composante, bloc, noms_variables), nb)
  d$variable <- factor(d$variable, levels = rev(d$variable))
  nm <- .noms(obj, noms)
  ggplot2::ggplot(d, ggplot2::aes(x = contribution, y = variable, fill = signe >= 0)) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::scale_fill_manual(values = c(`TRUE` = T$commune, `FALSE` = T$melange), guide = "none") +
    ggplot2::labs(x = .t("contribution", langue), y = NULL,
                  title = paste0("v", composante, " ", .netiq(obj$etiquettes[[composante]], nm, langue), " ; ", .t("signe", langue))) +
    .theme_gg() + ggplot2::theme(panel.grid.major.x = ggplot2::element_line(colour = T$grille, linewidth = 0.3),
                                 panel.grid.major.y = ggplot2::element_blank())
}

graph_stabilite <- function(stab, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_stabilite(stab, langue)
  parts <- unique(d$partie)
  d$partie_dim <- factor(paste0(d$partie, " (", d$dimension, ")"), levels = rev(unique(paste0(d$partie, " (", d$dimension, ")"))))
  d$critere <- factor(d$critere, levels = c("type", "etiquette", "strict"))
  lab <- c(strict = .t("strict", langue), etiquette = .t("etiquette", langue), type = .t("type", langue))
  ggplot2::ggplot(d, ggplot2::aes(x = 100 * frequence, y = partie_dim, fill = critere)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8), width = 0.75) +
    ggplot2::scale_fill_manual(values = c(strict = T$seq[1], etiquette = T$seq[2], type = T$seq[3]), labels = lab,
                               breaks = c("strict", "etiquette", "type"), name = .t("critere", langue)) +
    ggplot2::scale_x_continuous(limits = c(0, 100)) +
    ggplot2::labs(x = .t("stab_axe", langue), y = NULL,
                  title = .t("stab_titre", langue, nb = stab$nb, p = round(100 * stab$motif_retrouve))) +
    .theme_gg() + ggplot2::theme(legend.position = "bottom", legend.title = ggplot2::element_text(),
                                 panel.grid.major.x = ggplot2::element_line(colour = T$grille, linewidth = 0.3),
                                 panel.grid.major.y = ggplot2::element_blank())
}

graph_sensibilite <- function(sens, facteur_reference = 1, langue = NULL) {
  .gg(); T <- .T()
  d <- donnees_sensibilite(sens, facteur_reference)
  lg <- if (is.null(langue)) CoPRes::langue() else langue
  if (lg == "en") d$motif <- gsub("résiduelle", "residual", d$motif)
  d$texte <- gsub("), (", ")\n(", d$motif, fixed = TRUE)
  fs <- sort(unique(d$facteur))
  ggplot2::ggplot(d, ggplot2::aes(x = factor(facteur, levels = fs), y = factor(R, levels = rev(sort(unique(R)))))) +
    ggplot2::geom_tile(ggplot2::aes(fill = different), colour = "white", linewidth = 1) +
    ggplot2::geom_text(ggplot2::aes(label = texte), size = (T$taille - 2.7) / 2.8) +
    ggplot2::scale_fill_manual(values = c(`TRUE` = "#cde2fb", `FALSE` = "#f2f1ee"), guide = "none") +
    ggplot2::scale_x_discrete(position = "top", labels = function(f)
      vapply(f, function(v) .t("seuil_facteur", langue, f = sub("[,.]?0+$", "", .nombre(as.numeric(v), 2, langue))), "")) +
    ggplot2::scale_y_discrete(labels = function(r) paste("R =", r)) +
    ggplot2::labs(x = NULL, y = NULL, title = .t("sens_titre", langue)) +
    ggplot2::theme_minimal(base_size = T$taille + 2) + ggplot2::theme(panel.grid = ggplot2::element_blank())
}

graph_bilan <- function(X = NULL, theta = NULL, R = NULL, methodes = c("CPCA", "ACOM", "ACCPS"), departs = 20,
                        donnees = NULL, langue = NULL) {
  .gg(); T <- .T()
  d <- if (is.null(donnees)) donnees_bilan(X, theta, R, methodes, departs) else donnees
  d$i <- seq_len(nrow(d))
  e <- rbind(data.frame(i = d$i - 0.18, valeur = d$lecture_directe, serie = "lecture_directe"),
             data.frame(i = d$i + 0.18, valeur = d$copres, serie = "copres"))
  lab <- c(lecture_directe = .t("lecture_directe", langue), copres = .t("copres", langue), borne = .t("borne", langue))
  ggplot2::ggplot() +
    ggplot2::geom_col(data = e, ggplot2::aes(x = i, y = valeur, fill = serie), width = 0.34) +
    ggplot2::geom_text(data = e, ggplot2::aes(x = i, y = valeur * 0.97, label = .nombre(valeur, 3, langue),
                                              colour = serie), vjust = 1, size = (T$taille - 2) / 2.8, show.legend = FALSE) +
    ggplot2::geom_segment(data = d, ggplot2::aes(x = i - 0.38, xend = i + 0.38, y = borne, yend = borne, linetype = "borne"),
                          colour = T$encre) +
    ggplot2::scale_fill_manual(values = c(lecture_directe = T$non_porteur, copres = T$commune), labels = lab,
                               breaks = c("lecture_directe", "copres")) +
    ggplot2::scale_colour_manual(values = c(lecture_directe = T$encre, copres = "white")) +
    ggplot2::scale_linetype_manual(values = c(borne = "solid"), labels = lab["borne"]) +
    ggplot2::scale_x_continuous(breaks = d$i, labels = d$methode) +
    ggplot2::labs(x = NULL, y = .t("bilan", langue)) + .theme_gg()
}
