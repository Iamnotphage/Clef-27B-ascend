
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_CLEF_GDN_QKV_NORM_H_
#define ACLNN_CLEF_GDN_QKV_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnClefGdnQkvNormGetWorkspaceSize
 * parameters :
 * x : required
 * eps : required
 * qOut : required
 * kOut : required
 * vOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClefGdnQkvNormGetWorkspaceSize(
    const aclTensor *x,
    double eps,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnClefGdnQkvNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClefGdnQkvNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
