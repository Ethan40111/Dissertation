library(dplyr)
library(survival)
library(survminer)

lung <- read.csv("lung2.csv")
names(lung)

#rename column names
lung <- lung %>%
  rename(
    TNM_7th         = `Derived.AJCC.Stage.Group..7th.ed..2010.2015.`,
    event           = `Vital.status.recode..study.cutoff.used.`,
    survival.months = `Survival.months`,
    TNM_6th         = `Derived.AJCC.Stage.Group..6th.ed..2004.2015.`,
    TNM_3rd         = `AJCC.stage.3rd.edition..1988.2003.`  # note underscore (no spaces)
)
#so I can trial and error datasets

#check size
nrow(lung)



lung <- lung %>%
  mutate(
    TNM = case_when(
      TNM_7th != "Blank(s)" & !is.na(TNM_7th) ~ TNM_7th,
      TNM_6th != "Blank(s)" & !is.na(TNM_6th) ~ TNM_6th,
      TNM_3rd != "Blank(s)" & !is.na(TNM_3rd) ~ TNM_3rd,
      TRUE ~ "0"
    )
  )

#remove rows where TNM = 0
lung <- lung %>%
  filter(!is.na(TNM), TNM != "0", TNM != "Blank(s)")
nrow(lung)


# convert TNM values to corresponding stages
lung <- lung %>%
  mutate(
    TNM = as.character(TNM),  # ensure character
    TNM_stage = case_when(
      TNM %in% c("10","11","12","13") ~ "I",
      TNM %in% c("20","21","22","23") ~ "II",
      TNM %in% c("31","32","33")      ~ "III",
      TNM %in% c("40","41","42")      ~ "IV",
      TNM %in% c("IIA","IIB")      ~ "II",
      TNM %in% c("IIIA","IIIB") ~ "III",
      TNM %in% c("IA","IB") ~ "I",
      TRUE                             ~ TNM
    )
  ) %>%
  # remove rows you don't want
  filter(!TNM %in% c("90","98","Blank(s)","0","OCCULT","UNK Stage", "Stage 0"),
         !is.na(TNM_stage))

#replace TNM with TNM_stage
lung$TNM <- lung$TNM_stage; lung$TNM_stage <- NULL


#make survival.months numeric


lung <- lung %>%
  mutate(
    # make sure it's character first (avoids factor-to-integer mistakes)
    survival.months = as.numeric(as.character(survival.months)),
    survival.year = survival.months / 12
  )

fit <- survfit(Surv(survival.year, event == "Dead") ~ TNM, data = lung)

n_strata <- if (!is.null(fit$strata)) length(fit$strata) else 1

# pick colors (choose one of the following lines)
cols <- grDevices::rainbow(n_strata)  

plot(fit,xlim=c(0,10),
     col = cols, lwd = 2, mark.time = FALSE,
     xlab = "Years since diagnosis",
     ylab = "Survival probability",
     main = "KM Plot Grouped by TNM Stage")

tnm_lvls <- sub("^TNM=", "", names(fit$strata))
n <- length(tnm_lvls)

legend(
  "topright", title = "TNM Stage",
  legend = tnm_lvls, col = cols, lty = 1, lwd = 2, bty = "n"
)

#Get 10 year survival probability - To write in report 
s <- summary(fit, times = 10)


points(rep(10, length(s$surv)), s$surv, pch = 16, col = cols)
text(10, s$surv, labels = paste0(round(100*s$surv,0), "%"),
     pos = 4, cex = 0.8, col = cols, xpd = NA)
s$surv

